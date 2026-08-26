"""管理会话内已确认的查询条件，避免受控工具仅依赖当前一句提问。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ConversationContext

EXPLICIT_DATE_RANGE_PATTERN = re.compile(
    r"(?P<year>20\d{2})\s*年\s*(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*日?"
    r"\s*(?:至|到|~|-)\s*(?:(?P<end_year>20\d{2})\s*年\s*)?"
    r"(?P<end_month>\d{1,2})\s*月\s*(?P<end_day>\d{1,2})\s*日?"
)

# 演示数据中已登记的活动期。只有用户明确提到活动名称时才会采用，显式日期始终优先。
ACTIVITY_PERIODS = {
    "618": (("618", "六一八"), date(2026, 6, 1), date(2026, 6, 20)),
    "七夕": (("七夕",), date(2026, 8, 10), date(2026, 8, 22)),
    "春季上新": (("春季上新",), date(2026, 3, 8), date(2026, 3, 14)),
}

METRIC_HINTS = (
    ("支付 GMV", ("gmv", "成交额", "销售额", "营业额", "支付金额")),
    ("支付订单数", ("订单量", "订单数", "支付订单", "成交订单", "销量")),
    ("访客数", ("访客", "流量", "uv", "浏览人数", "访问量")),
    ("退款订单数", ("退款数", "退款订单", "售后订单")),
    ("支付转化率", ("转化率", "支付转化", "成交转化", "cvr")),
    ("客单价", ("客单价", "平均订单金额", "aov")),
    ("退款率", ("退款率", "退货率", "售后率")),
)


@dataclass(frozen=True)
class ConversationContextSnapshot:
    """一次会话中可安全继承的、仅来自用户明确输入的结构化条件。"""

    activity: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    metric_hints: tuple[str, ...] = ()
    knowledge_group: str | None = None
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
        if self.knowledge_group:
            parts.append(f"资料分组={self.knowledge_group}")
        if self.analysis_goal:
            parts.append(f"目标={self.analysis_goal}")
        return "；".join(parts)

    @property
    def retrieval_question(self) -> str:
        """将确认条件作为受控补充，而非伪造为新的用户提问。"""
        return self.display


@dataclass(frozen=True)
class ContextBuildResult:
    """本轮合并后的条件及其更新、继承轨迹。"""

    snapshot: ConversationContextSnapshot
    updated_fields: tuple[str, ...]
    inherited_fields: tuple[str, ...]

    @property
    def audit_actions(self) -> tuple[str, ...]:
        actions: list[str] = []
        if self.updated_fields:
            actions.append(f"本轮确认/覆盖：{'、'.join(self.updated_fields)}")
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
    selected_knowledge_group: str | None,
) -> ContextBuildResult:
    """读取会话状态，合并本轮明确条件，并在用户消息入库后立即持久化。"""
    record = await session.get(ConversationContext, conversation_id)
    previous = _snapshot_from_record(record)
    result = build_context_snapshot(
        previous, question, selected_knowledge_group, source_message_id
    )
    if record is None:
        record = ConversationContext(conversation_id=conversation_id)
        session.add(record)
    _apply_snapshot(record, result.snapshot)
    return result


def build_context_snapshot(
    previous: ConversationContextSnapshot,
    question: str,
    selected_knowledge_group: str | None,
    source_message_id: str,
) -> ContextBuildResult:
    """以确定性规则合并条件；未出现的字段只继承，不由模型推测。"""
    values = {
        "activity": previous.activity,
        "start_date": previous.start_date,
        "end_date": previous.end_date,
        "metric_hints": previous.metric_hints,
        "knowledge_group": previous.knowledge_group,
        "analysis_goal": previous.analysis_goal,
    }
    sources = dict(previous.field_sources or {})
    explicit = _extract_explicit_conditions(question, selected_knowledge_group)
    updated_fields: list[str] = []
    for field_name, value in explicit.items():
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
            knowledge_group=values["knowledge_group"],
            analysis_goal=values["analysis_goal"],
            field_sources=sources,
        ),
        updated_fields=tuple(dict.fromkeys(updated_fields)),
        inherited_fields=inherited_fields,
    )


def build_retrieval_question(user_input: str, snapshot: ConversationContextSnapshot) -> str:
    """为路由和受控工具补充条件；原始问题仍单独保留给回答生成层。"""
    if not snapshot.retrieval_question:
        return user_input
    return f"{user_input}\n\n已确认会话查询条件（仅用于限定本轮检索范围）：{snapshot.retrieval_question}"


def _extract_explicit_conditions(
    question: str, selected_knowledge_group: str | None
) -> dict[str, object]:
    lowered = question.lower()
    conditions: dict[str, object] = {}
    for activity, (aliases, start_date, end_date) in ACTIVITY_PERIODS.items():
        if any(alias.lower() in lowered for alias in aliases):
            conditions["activity"] = activity
            conditions["start_date"] = start_date
            conditions["end_date"] = end_date
            break
    explicit_period = _parse_explicit_date_range(question)
    if explicit_period is not None:
        conditions["start_date"], conditions["end_date"] = explicit_period
    metric_hints = tuple(
        name
        for name, aliases in METRIC_HINTS
        if any(alias.lower() in lowered for alias in aliases)
    )
    if metric_hints:
        conditions["metric_hints"] = metric_hints
    if selected_knowledge_group:
        conditions["knowledge_group"] = selected_knowledge_group.strip()
    if "复盘" in question:
        conditions["analysis_goal"] = "活动复盘"
    elif "分析" in question:
        conditions["analysis_goal"] = "经营分析"
    return conditions


def _parse_explicit_date_range(question: str) -> tuple[date, date] | None:
    match = EXPLICIT_DATE_RANGE_PATTERN.search(question)
    if match is None:
        return None
    try:
        start_date = date(int(match["year"]), int(match["month"]), int(match["day"]))
        end_date = date(
            int(match["end_year"] or match["year"]),
            int(match["end_month"]),
            int(match["end_day"]),
        )
    except ValueError:
        return None
    return (start_date, end_date) if start_date <= end_date else None


def _snapshot_from_record(record: ConversationContext | None) -> ConversationContextSnapshot:
    if record is None:
        return ConversationContextSnapshot()
    return ConversationContextSnapshot(
        activity=record.activity,
        start_date=record.start_date,
        end_date=record.end_date,
        metric_hints=tuple(record.metric_hints or ()),
        knowledge_group=record.knowledge_group,
        analysis_goal=record.analysis_goal,
        field_sources=dict(record.field_sources or {}),
    )


def _apply_snapshot(record: ConversationContext, snapshot: ConversationContextSnapshot) -> None:
    record.activity = snapshot.activity
    record.start_date = snapshot.start_date
    record.end_date = snapshot.end_date
    record.metric_hints = list(snapshot.metric_hints)
    record.knowledge_group = snapshot.knowledge_group
    record.analysis_goal = snapshot.analysis_goal
    record.field_sources = dict(snapshot.field_sources or {})


def _field_label(field_name: str) -> str:
    return {
        "activity": "活动",
        "start_date": "时间范围",
        "end_date": "时间范围",
        "metric_hints": "指标",
        "knowledge_group": "资料分组",
        "analysis_goal": "分析目标",
    }[field_name]


def _unique_labels(field_names: Iterable[str]) -> tuple[str, ...]:
    """将开始、结束日期等同属一个业务字段的更新合并为一条审计说明。"""
    return tuple(dict.fromkeys(_field_label(field_name) for field_name in field_names))
