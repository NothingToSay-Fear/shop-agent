"""管理会话内已确认的查询条件，避免受控工具仅依赖当前一句提问。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AgentRun, ConversationContext, Message, MetricDefinition
from app.services.date_ranges import DEMO_DATA_YEAR, parse_explicit_date_range
from app.services.metric_rag import METRIC_DEFINITION_SEEDS

# 演示数据中已登记的活动期。只有用户明确提到活动名称时才会采用，显式日期始终优先。
ACTIVITY_PERIODS = {
    "618": (("618", "六一八"), date(DEMO_DATA_YEAR, 6, 1), date(DEMO_DATA_YEAR, 6, 20)),
    "七夕": (("七夕",), date(DEMO_DATA_YEAR, 8, 10), date(DEMO_DATA_YEAR, 8, 22)),
    "春季上新": (("春季上新",), date(DEMO_DATA_YEAR, 3, 8), date(DEMO_DATA_YEAR, 3, 14)),
}
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
    "knowledge_group": ("清除资料分组", "取消资料分组", "不限定资料分组", "全部资料分组"),
    "analysis_goal": ("清除分析目标", "不限定分析目标"),
}


@dataclass(frozen=True)
class RecentTurnSummary:
    """仅携带上一轮已完成回答的受限摘要，用于处理“刚才/上一步”等指代。"""

    run_id: str
    answer_excerpt: str
    reference_ids: tuple[str, ...]

    @property
    def display(self) -> str:
        references = "、".join(self.reference_ids) if self.reference_ids else "无"
        return f"上一轮结论摘要：{self.answer_excerpt}\n上一轮引用 ID：{references}\n来源运行：{self.run_id}"


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
    recent_turn: RecentTurnSummary | None = None

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

    @property
    def generation_context(self) -> str:
        """生成层可见的最小上下文；最近结论绝不参与路由或工具入参。"""
        parts = [f"已确认查询条件：{self.display or '无'}"]
        if self.recent_turn is not None:
            parts.append(
                "以下是上一轮已完成回答的受限摘要，仅用于理解指代，"
                "不能替代本轮工具依据：\n" + self.recent_turn.display
            )
        return "\n\n".join(parts)

    def as_audit_snapshot(self) -> dict[str, object]:
        """保存本轮实际采用的结构化字段，便于后续准确复盘。"""
        return {
            "activity": self.activity,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "metric_hints": list(self.metric_hints),
            "knowledge_group": self.knowledge_group,
            "analysis_goal": self.analysis_goal,
            "field_sources": dict(self.field_sources or {}),
            "recent_turn": (
                {
                    "run_id": self.recent_turn.run_id,
                    "reference_ids": list(self.recent_turn.reference_ids),
                }
                if self.recent_turn
                else None
            ),
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
    selected_knowledge_group: str | None,
) -> ContextBuildResult:
    """读取会话状态，合并本轮明确条件，并在用户消息入库后立即持久化。"""
    record = await session.get(ConversationContext, conversation_id)
    previous = _snapshot_from_record(record)
    metric_hints = await _load_metric_hints(session)
    result = build_context_snapshot(
        previous, question, selected_knowledge_group, source_message_id, metric_hints
    )
    recent_turn = await _load_recent_turn_summary(session, conversation_id)
    result = replace(result, snapshot=replace(result.snapshot, recent_turn=recent_turn))
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
    metric_hints: Iterable[MetricHint] = DEFAULT_METRIC_HINTS,
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
    explicit = _extract_explicit_conditions(question, selected_knowledge_group, metric_hints)
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
            knowledge_group=values["knowledge_group"],
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


def _extract_explicit_conditions(
    question: str, selected_knowledge_group: str | None, metric_hints: Iterable[MetricHint]
) -> dict[str, object]:
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
    if selected_knowledge_group:
        conditions["knowledge_group"] = selected_knowledge_group.strip()
    if "复盘" in question:
        conditions["analysis_goal"] = "活动复盘"
    elif "分析" in question:
        conditions["analysis_goal"] = "经营分析"
    return conditions


def _extract_clear_conditions(question: str) -> dict[str, object]:
    fields = ("activity", "start_date", "end_date", "metric_hints", "knowledge_group", "analysis_goal")
    conditions: dict[str, object] = {}
    if any(pattern in question for pattern in _CLEAR_ALL_PATTERNS):
        return {
            field_name: (() if field_name == "metric_hints" else None) for field_name in fields
        }
    if any(pattern in question for pattern in _CLEAR_FIELD_PATTERNS["activity"]):
        conditions["activity"] = None
    if any(pattern in question for pattern in _CLEAR_FIELD_PATTERNS["time_range"]):
        conditions.update({"activity": None, "start_date": None, "end_date": None})
    for field_name in ("metric_hints", "knowledge_group", "analysis_goal"):
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


async def _load_recent_turn_summary(
    session: AsyncSession, conversation_id: str
) -> RecentTurnSummary | None:
    """从消息主存读取最近一条已完成回答，限制长度后仅在本轮生成阶段使用。"""
    row = (
        await session.execute(
            select(AgentRun, Message.content)
            .join(Message, AgentRun.agent_message_id == Message.id)
            .where(AgentRun.conversation_id == conversation_id, AgentRun.status == "completed")
            .order_by(AgentRun.completed_at.desc())
            .limit(1)
        )
    ).first()
    if row is None:
        return None
    run, answer = row
    return RecentTurnSummary(
        run_id=run.id,
        answer_excerpt=_truncate_summary(answer),
        reference_ids=tuple(run.reference_ids or ()),
    )


def _truncate_summary(answer: str, limit: int = 500) -> str:
    normalized = re.sub(r"\s+", " ", answer).strip()
    return normalized if len(normalized) <= limit else f"{normalized[:limit]}…"


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
