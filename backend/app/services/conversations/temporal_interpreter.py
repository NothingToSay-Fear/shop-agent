"""以受限 LLM 将用户时间表达规整为后端可校验的日期范围。"""

from __future__ import annotations

import json
import logging
import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.services.models.llm_factory import LLMProviderFactory
from app.models import DailyMetric
from app.services.analytics.business_dates import current_business_date
from app.services.analytics.activity_periods import resolve_activity_periods
from app.services.analytics.date_ranges import parse_explicit_date_ranges
from app.services.models.embeddings import embed_texts

TemporalStatus = Literal["no_time", "resolved", "clarify", "forecast"]
TemporalSource = Literal["llm", "local_gate", "unavailable", "invalid"]

logger = logging.getLogger(__name__)
_MAX_PERIODS = 4
_NO_TIME_MINIMUM_SCORE = 0.58
_NO_TIME_MINIMUM_MARGIN = 0.08
_TEMPORAL_PROTOTYPES = (
    "查询、比较或分析某个日期、天、周、月、季度、年度、活动期或相对时间范围的数据",
    "今天、昨天、本自然周、上周、本月、去年同期、最近一段时间等时间表达",
    "继续查看刚才相同时间范围内的另一个指标、解释原因或给出建议，不改变时间条件",
    "询问资料规则、经营指标含义、归因结论或建议，不要求改变或指定时间范围",
    "再看看订单量、访客、转化率等另一个指标，沿用当前已经确认的时间范围",
    "继续分析刚才的数据、解释下降原因或给出建议，不新增日期或统计周期",
)
_prototype_embedding_cache: dict[str, tuple[list[float], ...]] = {}
_prototype_embedding_lock = asyncio.Lock()


@dataclass(frozen=True)
class TemporalResolution:
    """LLM 时间输出经后端校验后的唯一形式。"""

    status: TemporalStatus
    periods: tuple[tuple[str, date, date], ...] = ()
    clarification: str | None = None
    source: TemporalSource = "llm"

    @property
    def resets_inherited_range(self) -> bool:
        """仅明确解析出的新范围才能覆盖既有范围。"""
        return self.status == "resolved"


_PROMPT = """你是电商运营问题的时间范围解析器。只输出 JSON，不能回答业务问题、不能输出 SQL。
你必须基于“当前服务日期”“最新可用业务数据日”和“既有任务上下文”（若有）理解任意中文时间表达，并把实际查询范围直接规整为 ISO 日期。

只允许以下 JSON 结构：
{"status":"no_time|resolved|clarify|forecast","periods":[{"label":"不超过40字","start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD"}],"clarification":"不超过160字"}

规则：
1. 没有时间条件时 status=no_time，periods=[]。
2. 可用实际数据回答时 status=resolved；periods 为 1 至 4 个闭区间，日期不得晚于最新可用业务数据日。
3. “本自然周/上自然周”等表达按用户问题和提供的业务规则规整；比较区间必须在 periods 中完整列出，不能只给一段。
4. 若业务周定义、实际/预测含义或范围无法确定，status=clarify，并给出简短追问；不要猜测。
5. 用户明确要求未来或预测数据时 status=forecast；本系统尚未启用预测，不能把未来日期放入 periods。
6. 不得输出额外字段、Markdown 或解释文字。"""


async def resolve_temporal_intent(
    session: AsyncSession,
    question: str,
    settings: Settings | None = None,
    task_context: str | None = None,
) -> TemporalResolution:
    """本地门控决定是否调用 LLM；后端只校验其规范化日期。"""
    active_settings = settings or get_settings()
    explicit_periods = parse_explicit_date_ranges(question)
    if explicit_periods:
        return TemporalResolution(
            "resolved",
            tuple((label, start_date, end_date) for label, (start_date, end_date) in explicit_periods),
            source="local_gate",
        )
    activity_periods = resolve_activity_periods(question)
    if activity_periods:
        return TemporalResolution("resolved", activity_periods, source="local_gate")
    if not await _needs_temporal_normalization(question, active_settings):
        return TemporalResolution("no_time", source="local_gate")
    latest_date = await session.scalar(
        select(func.max(DailyMetric.metric_date)).where(
            DailyMetric.source == "demo",
            DailyMetric.metric_date <= current_business_date(),
        )
    )
    if not active_settings.llm_enabled:
        return TemporalResolution(
            "clarify",
            clarification="当前未配置时间解析模型，无法可靠识别查询时间。请补充明确的起止日期范围。",
            source="unavailable",
        )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = await LLMProviderFactory.create(active_settings, temperature=0).ainvoke(
            [
                SystemMessage(content=_PROMPT),
                HumanMessage(
                    content=(
                        f"当前服务日期={current_business_date()}\n"
                        f"最新可用业务数据日={latest_date or '无'}\n"
                        "业务周规则：若用户明确说“自然周”，周一为一周起始；"
                        "本周实际值截至最新可用业务数据日，上周比较时取同天数同期。\n"
                        f"既有待续任务上下文={task_context or '无'}\n"
                        f"用户问题={question[:1200]}"
                    )
                ),
            ]
        )
    except Exception:
        logger.warning("temporal_normalization_llm_failed", exc_info=True)
        return TemporalResolution(
            "clarify",
            clarification="时间范围解析暂时不可用，请补充明确的起止日期范围后重试。",
            source="unavailable",
        )
    return _validate_llm_resolution(response.content, latest_date)


async def _needs_temporal_normalization(question: str, settings: Settings) -> bool:
    """仅在本地语义模型高度确认“未改变时间”时跳过外部 LLM。"""
    question_vectors = await embed_texts([question], settings)
    if not question_vectors:
        # 本地模型不可用时不冒漏检风险，交由时间 LLM 判断。
        return True
    prototypes = await _get_temporal_prototype_embeddings(settings)
    if prototypes is None:
        return True
    time_score = max(_cosine_similarity(question_vectors[0], item) for item in prototypes[:2])
    no_time_score = max(_cosine_similarity(question_vectors[0], item) for item in prototypes[2:])
    return _needs_llm_from_scores(time_score, no_time_score)


def _needs_llm_from_scores(time_score: float, no_time_score: float) -> bool:
    """不确定即调用 LLM；只跳过高置信度且有明显分差的无时间轮次。"""
    return not (
        no_time_score >= _NO_TIME_MINIMUM_SCORE
        and no_time_score - time_score >= _NO_TIME_MINIMUM_MARGIN
    )


async def _get_temporal_prototype_embeddings(settings: Settings) -> tuple[list[float], ...] | None:
    cache_key = (
        f"{settings.local_embedding_model_id}:{settings.local_embedding_model_path}:"
        f"{settings.local_embedding_device}"
    )
    cached = _prototype_embedding_cache.get(cache_key)
    if cached is not None:
        return cached
    async with _prototype_embedding_lock:
        cached = _prototype_embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        embeddings = await embed_texts(list(_TEMPORAL_PROTOTYPES), settings)
        if embeddings is None:
            return None
        cached = tuple(embeddings)
        _prototype_embedding_cache[cache_key] = cached
        return cached


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


def _validate_llm_resolution(content: object, latest_date: date | None) -> TemporalResolution:
    """校验 LLM 日期输出，禁止未经验证的范围进入受控 SQL。"""
    try:
        payload = json.loads(_json_text(content))
    except (TypeError, json.JSONDecodeError):
        return _invalid_resolution("时间解析结果格式无效，请补充明确的起止日期范围。")
    if not isinstance(payload, dict):
        return _invalid_resolution("时间解析结果格式无效，请补充明确的起止日期范围。")
    status = payload.get("status")
    if status == "no_time":
        return TemporalResolution("no_time")
    if status == "clarify":
        return TemporalResolution("clarify", clarification=_clarification(payload))
    if status == "forecast":
        return TemporalResolution(
            "forecast",
            clarification="当前仅支持实际经营数据查询，尚未启用预测能力。请改为查询实际日期范围。",
        )
    if status != "resolved":
        return _invalid_resolution("无法确认查询时间，请补充明确的起止日期范围。")
    raw_periods = payload.get("periods")
    if not isinstance(raw_periods, list) or not 1 <= len(raw_periods) <= _MAX_PERIODS:
        return _invalid_resolution("时间范围数量无效，请补充不超过 4 个明确日期区间。")
    periods: list[tuple[str, date, date]] = []
    seen: set[tuple[date, date]] = set()
    for index, item in enumerate(raw_periods, start=1):
        if not isinstance(item, dict):
            return _invalid_resolution("时间范围格式无效，请补充明确的起止日期范围。")
        try:
            start = date.fromisoformat(str(item.get("start_date", "")))
            end = date.fromisoformat(str(item.get("end_date", "")))
        except ValueError:
            return _invalid_resolution("日期必须为 YYYY-MM-DD 格式，请补充明确范围。")
        if start > end:
            return _invalid_resolution("日期起止顺序无效，请补充明确范围。")
        if latest_date is None or end > latest_date:
            return _invalid_resolution(
                f"实际经营数据当前仅覆盖至 {latest_date or '暂无数据'}，请调整查询范围或明确预测需求。"
            )
        if (start, end) in seen:
            continue
        seen.add((start, end))
        label = " ".join(str(item.get("label") or f"区间 {index}").split())[:40]
        periods.append((label, start, end))
    if not periods:
        return _invalid_resolution("时间范围重复或无效，请补充明确范围。")
    return TemporalResolution("resolved", tuple(periods))


def _json_text(content: object) -> str:
    """兼容部分模型返回的 Markdown JSON 包裹。"""
    return str(content).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _clarification(payload: dict[str, object]) -> str:
    value = " ".join(str(payload.get("clarification") or "").split())[:160]
    return value or "请确认要查询的日期范围，或说明是实际累计还是预测数据。"


def _invalid_resolution(message: str) -> TemporalResolution:
    return TemporalResolution("clarify", clarification=message, source="invalid")
