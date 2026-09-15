"""管理会话内已确认的查询条件，避免受控工具仅依赖当前一句提问。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ConversationContext, MetricDefinition
from app.services.activity_periods import ACTIVITY_PERIODS
from app.services.date_ranges import parse_explicit_date_range
from app.services.metric_rag import METRIC_DEFINITION_SEEDS

MetricHint = tuple[str, tuple[str, ...]]
DEFAULT_METRIC_HINTS: tuple[MetricHint, ...] = tuple(
    (str(item["name"]), tuple([str(item["name"]), *[str(alias) for alias in item["aliases"]]]))
    for item in METRIC_DEFINITION_SEEDS
)
_CLEAR_ALL_PATTERNS = ("清除会话条件", "重置会话条件", "清空会话条件", "不继承上文")
_CLEAR_FIELD_PATTERNS = {
    "activity": ("清除活动", "取消活动", "不限定活动"),
    "time_range": ("清除时间", "清除日期", "不限定时间", "不看活动期"),
    "metric_hints": ("清除指标", "不限定指标"),
    "analysis_goal": ("清除分析目标", "不限定分析目标"),
}


@dataclass(frozen=True)
class ConversationContextSnapshot:
    """一次会话中可安全继承的、仅来自用户明确输入的结构化条件。"""

    activity: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    metric_hints: tuple[str, ...] = ()
    analysis_goal: str | None = None
    field_sources: dict[str, str] | None = None

    @property
    def display(self) -> str:
        """返回可展示且可传给工具的条件摘要，不包含历史问题原文。"""
        parts: list[str] = []
        if self.activity:
            parts.append(f"活动={self.activity}")
        if self.start_date and self.end_date:
            parts.append(f"时间={self.start_date} 至 {self.end_date}")
        if self.metric_hints:
            parts.append(f"指标={'、'.join(self.metric_hints)}")
        if self.analysis_goal:
            parts.append(f"目标={self.analysis_goal}")
        return "；".join(parts)

    @property
    def retrieval_question(self) -> str:
        """将确认条件作为受控补充，而非伪造为新的用户提问。"""
        return self.display

    @property
    def generation_context(self) -> str:
        """生成层可见的最小结构化条件；不混入会话原文。"""
        return f"已确认查询条件：{self.display or '无'}"

    def as_audit_snapshot(self) -> dict[str, object]:
        """保存本轮实际采用的结构化字段，便于后续准确复盘。"""
        return {
            "activity": self.activity,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "metric_hints": list(self.metric_hints),
            "analysis_goal": self.analysis_goal,
            "field_sources": dict(self.field_sources or {}),
        }


@dataclass(frozen=True)
class ContextBuildResult:
    """本轮合并后的条件及其更新、继承轨迹。"""

    snapshot: ConversationContextSnapshot
    updated_fields: tuple[str, ...]
    inherited_fields: tuple[str, ...]
    cleared_fields: tuple[str, ...] = ()

    @property
    def audit_actions(self) -> tuple[str, ...]:
        actions: list[str] = []
        if self.updated_fields:
            actions.append(f"本轮确认/覆盖：{'、'.join(self.updated_fields)}")
        if self.cleared_fields:
            actions.append(f"本轮清除：{'、'.join(self.cleared_fields)}")
        if self.inherited_fields:
            actions.append(f"继承已确认条件：{'、'.join(self.inherited_fields)}")
        if not actions:
            actions.append("本轮未识别到可继承的明确条件")
        return tuple(actions)

    @property
    def audit_summary(self) -> str:
        return f"有效条件：{self.snapshot.display or '无'}；{'；'.join(self.audit_actions)}"


async def build_and_persist_context(
    session: AsyncSession,
    conversation_id: str,
    source_message_id: str,
    question: str,
) -> ContextBuildResult:
    """读取会话状态，合并本轮明确条件，并在用户消息入库后立即持久化。"""
    record = await session.get(ConversationContext, conversation_id)
    previous = _snapshot_from_record(record)
    metric_hints = await _load_metric_hints(session)
    result = build_context_snapshot(
        previous, question, source_message_id, metric_hints
    )
    if record is None:
        record = ConversationContext(conversation_id=conversation_id)
        session.add(record)
    _apply_snapshot(record, result.snapshot)
    return result


def build_context_snapshot(
    previous: ConversationContextSnapshot,
    question: str,
    source_message_id: str,
    metric_hints: Iterable[MetricHint] = DEFAULT_METRIC_HINTS,
) -> ContextBuildResult:
    """以确定性规则合并条件；未出现的字段只继承，不由模型推测。"""
    values = {
        "activity": previous.activity,
        "start_date": previous.start_date,
        "end_date": previous.end_date,
        "metric_hints": previous.metric_hints,
        "analysis_goal": previous.analysis_goal,
    }
    sources = dict(previous.field_sources or {})
    explicit = _extract_explicit_conditions(question, metric_hints)
    updated_fields: list[str] = []
    cleared_fields: list[str] = []
    for field_name, value in explicit.items():
        is_clear = value is None or (field_name == "metric_hints" and value == ())
        if is_clear:
            if values[field_name] is not None:
                cleared_fields.append(_field_label(field_name))
            values[field_name] = value
            sources.pop(field_name, None)
        else:
            if values[field_name] != value:
                values[field_name] = value
            sources[field_name] = source_message_id
            updated_fields.append(_field_label(field_name))

    inherited_fields = _unique_labels(
        field_name for field_name, value in values.items() if value and field_name not in explicit
    )
    return ContextBuildResult(
        snapshot=ConversationContextSnapshot(
            activity=values["activity"],
            start_date=values["start_date"],
            end_date=values["end_date"],
            metric_hints=values["metric_hints"],
            analysis_goal=values["analysis_goal"],
            field_sources=sources,
        ),
        updated_fields=tuple(dict.fromkeys(updated_fields)),
        inherited_fields=inherited_fields,
        cleared_fields=tuple(dict.fromkeys(cleared_fields)),
    )


def build_retrieval_question(user_input: str, snapshot: ConversationContextSnapshot) -> str:
    """为路由和受控工具补充条件；原始问题仍单独保留给回答生成层。"""
    if not snapshot.retrieval_question:
        return user_input
    return f"{user_input}\n\n已确认会话查询条件（仅用于限定本轮检索范围）：{snapshot.retrieval_question}"


def _extract_explicit_conditions(question: str, metric_hints: Iterable[MetricHint]) -> dict[str, object]:
    lowered = question.lower()
    conditions = _extract_clear_conditions(lowered)
    activity_detected = False
    for activity, (aliases, start_date, end_date) in ACTIVITY_PERIODS.items():
        if any(alias.lower() in lowered for alias in aliases):
            conditions["activity"] = activity
            conditions["start_date"] = start_date
            conditions["end_date"] = end_date
            activity_detected = True
            break
    explicit_period = parse_explicit_date_range(question)
    if explicit_period is not None:
        if not activity_detected:
            # 新日期范围不应继续携带旧活动标签，避免“618 + 七夕日期”这种矛盾状态。
            conditions["activity"] = None
        conditions["start_date"], conditions["end_date"] = explicit_period
    matched_metric_hints = tuple(
        name
        for name, aliases in metric_hints
        if any(alias.lower() in lowered for alias in aliases)
    )
    if matched_metric_hints:
        conditions["metric_hints"] = matched_metric_hints
    if "复盘" in question:
        conditions["analysis_goal"] = "活动复盘"
    elif "分析" in question:
        conditions["analysis_goal"] = "经营分析"
    return conditions


def _extract_clear_conditions(question: str) -> dict[str, object]:
    fields = ("activity", "start_date", "end_date", "metric_hints", "analysis_goal")
    conditions: dict[str, object] = {}
    if any(pattern in question for pattern in _CLEAR_ALL_PATTERNS):
        return {
            field_name: (() if field_name == "metric_hints" else None) for field_name in fields
        }
    if any(pattern in question for pattern in _CLEAR_FIELD_PATTERNS["activity"]):
        conditions["activity"] = None
    if any(pattern in question for pattern in _CLEAR_FIELD_PATTERNS["time_range"]):
        conditions.update({"activity": None, "start_date": None, "end_date": None})
    for field_name in ("metric_hints", "analysis_goal"):
        if any(pattern in question for pattern in _CLEAR_FIELD_PATTERNS[field_name]):
            conditions[field_name] = () if field_name == "metric_hints" else None
    return conditions


async def _load_metric_hints(session: AsyncSession) -> tuple[MetricHint, ...]:
    """复用指标定义表中的名称和别名，避免上下文层维护第二份指标词典。"""
    definitions = list(
        await session.scalars(select(MetricDefinition).where(MetricDefinition.enabled.is_(True)))
    )
    if not definitions:
        return DEFAULT_METRIC_HINTS
    return tuple((item.name, tuple([item.name, *item.aliases])) for item in definitions)


def _snapshot_from_record(record: ConversationContext | None) -> ConversationContextSnapshot:
    if record is None:
        return ConversationContextSnapshot()
    return ConversationContextSnapshot(
        activity=record.activity,
        start_date=record.start_date,
        end_date=record.end_date,
        metric_hints=tuple(record.metric_hints or ()),
        analysis_goal=record.analysis_goal,
        field_sources=dict(record.field_sources or {}),
    )


def _apply_snapshot(record: ConversationContext, snapshot: ConversationContextSnapshot) -> None:
    record.activity = snapshot.activity
    record.start_date = snapshot.start_date
    record.end_date = snapshot.end_date
    record.metric_hints = list(snapshot.metric_hints)
    record.analysis_goal = snapshot.analysis_goal
    record.field_sources = dict(snapshot.field_sources or {})


def _field_label(field_name: str) -> str:
    return {
        "activity": "活动",
        "start_date": "时间范围",
        "end_date": "时间范围",
        "metric_hints": "指标",
        "analysis_goal": "分析目标",
    }[field_name]


def _unique_labels(field_names: Iterable[str]) -> tuple[str, ...]:
    """将开始、结束日期等同属一个业务字段的更新合并为一条审计说明。"""
    return tuple(dict.fromkeys(_field_label(field_name) for field_name in field_names))
