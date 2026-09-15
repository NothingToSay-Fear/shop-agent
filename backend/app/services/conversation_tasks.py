"""管理可跨轮补充的会话任务，并在执行前产出完整、可审计的问题。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ConversationTask
from app.services.intent_router import RetrievalRoute
from app.services.task_interpreter import (
    TaskRelationshipDecision,
    interpret_task_relationship,
)
from app.services.temporal_interpreter import TemporalResolution, resolve_temporal_intent

TASK_WAITING_CLARIFICATION = "waiting_clarification"
TASK_READY = "ready"
TASK_COMPLETED = "completed"
TASK_SUPERSEDED = "superseded"
TASK_CANCELLED = "cancelled"

TASK_METRIC_QUERY = "metric_query"
TASK_METRIC_COMPARISON = "metric_comparison"
TASK_KNOWLEDGE_QA = "knowledge_qa"
TASK_HYBRID_ANALYSIS = "hybrid_analysis"
TASK_REVIEW = "review"

_METRIC_MARKERS = ("gmv", "成交额", "销售额", "订单", "访客", "uv", "转化率", "客单价", "退款率")
_COMPARISON_MARKERS = ("对比", "相比", "环比", "同比", "较")
_KNOWLEDGE_MARKERS = ("资料", "文档", "规则", "玩法", "手册", "复盘", "文件")
_FRAME_TEXT_LIMIT = 1200
_MAX_SUPPLEMENTS = 3


@dataclass(frozen=True)
class ConversationTaskTurn:
    """本轮应继续执行任务，还是先向用户索取缺失条件。"""

    task: ConversationTask
    effective_question: str
    route_override: RetrievalRoute | None = None
    clarification: str | None = None
    temporal_resolution: TemporalResolution = TemporalResolution("no_time")

    @property
    def requires_clarification(self) -> bool:
        return self.clarification is not None


async def prepare_conversation_task(
    session: AsyncSession,
    conversation_id: str,
    source_message_id: str,
    question: str,
) -> ConversationTaskTurn:
    """识别新任务、补充条件和显式续问；只让完整任务进入 RAG 执行链路。"""
    pending = await _latest_task(session, conversation_id, (TASK_WAITING_CLARIFICATION, TASK_READY))
    completed = await _latest_task(session, conversation_id, (TASK_COMPLETED,))
    temporal_resolution = await resolve_temporal_intent(
        session,
        question,
        task_context=_temporal_task_context(pending or completed),
    )
    if pending is not None:
        decision = await interpret_task_relationship(pending, question)
        _record_task_interpretation(pending, source_message_id, decision)
        if decision.relation == "cancel":
            pending.status = TASK_CANCELLED
        elif decision.relation in {"continue", "revise"} and temporal_resolution.resets_inherited_range:
            # 当前轮的时间语义由受限解析确认，不能把旧任务的等待条件带入新任务。
            pending.status = TASK_SUPERSEDED
            return await _create_new_task(
                session,
                conversation_id,
                source_message_id,
                question,
                temporal_resolution,
                continued_from=pending,
            )
        elif decision.relation in {"continue", "revise"}:
            return _waiting_turn(pending)
        else:
            pending.status = TASK_SUPERSEDED

    if completed is not None:
        decision = await interpret_task_relationship(completed, question)
        _record_task_interpretation(completed, source_message_id, decision)
        if decision.relation in {"continue", "revise"} and temporal_resolution.resets_inherited_range:
            # 当前轮已有新的受控时间范围，必须重建任务，不能复用上一轮日期。
            completed.status = TASK_SUPERSEDED
            return await _create_new_task(
                session,
                conversation_id,
                source_message_id,
                question,
                temporal_resolution,
                continued_from=completed,
            )
        elif decision.relation in {"continue", "revise"}:
            return _continue_completed_task(completed, source_message_id, question)

    return await _create_new_task(
        session, conversation_id, source_message_id, question, temporal_resolution
    )


async def complete_conversation_task(
    task: ConversationTask, route: RetrievalRoute | None
) -> None:
    """只在本轮 Agent 正常完成后关闭任务，并保存实际采用的路由供后续续问复用。"""
    task.status = TASK_COMPLETED
    task.completed_at = datetime.now(UTC)
    if route is not None:
        task.route_mode = route.mode


async def cancel_open_conversation_tasks(session: AsyncSession, conversation_id: str) -> None:
    """清除会话条件时同步取消尚未完成的任务，避免旧任务误接收后续消息。"""
    tasks = list(
        await session.scalars(
            select(ConversationTask).where(
                ConversationTask.conversation_id == conversation_id,
                ConversationTask.status.in_((TASK_WAITING_CLARIFICATION, TASK_READY)),
            )
        )
    )
    for task in tasks:
        task.status = TASK_CANCELLED


async def _create_new_task(
    session: AsyncSession,
    conversation_id: str,
    source_message_id: str,
    question: str,
    temporal_resolution: TemporalResolution,
    continued_from: ConversationTask | None = None,
) -> ConversationTaskTurn:
    previous_frame = dict(continued_from.task_frame or {}) if continued_from else {}
    task_type = continued_from.task_type if continued_from else _infer_task_type(question)
    missing_slots = ["时间范围或数据类型"] if temporal_resolution.clarification else []
    task_frame: dict[str, object] = {
        "base_question": str(previous_frame.get("base_question") or _truncate(question)),
        "supplements": _append_supplement(previous_frame, question) if continued_from else [],
    }
    task_frame["temporal_resolution"] = {
        "status": temporal_resolution.status,
        "source": temporal_resolution.source,
    }
    if temporal_resolution.periods:
        resolved_periods = [
            {"label": label, "start_date": start.isoformat(), "end_date": end.isoformat()}
            for label, start, end in temporal_resolution.periods
        ]
        task_frame["resolved_periods"] = resolved_periods
    if temporal_resolution.clarification:
        task_frame["time_clarification"] = temporal_resolution.clarification
    task = ConversationTask(
        conversation_id=conversation_id,
        task_type=task_type,
        status=TASK_WAITING_CLARIFICATION if missing_slots else TASK_READY,
        route_mode="metrics" if task_type == TASK_METRIC_COMPARISON else None,
        task_frame=task_frame,
        missing_slots=missing_slots,
        source_message_ids=[source_message_id],
    )
    session.add(task)
    await session.flush()
    if missing_slots:
        return _waiting_turn(task, temporal_resolution)
    return ConversationTaskTurn(
        task, _effective_question(task), _route_override(task), temporal_resolution=temporal_resolution
    )


def _continue_completed_task(
    task: ConversationTask, source_message_id: str, question: str
) -> ConversationTaskTurn:
    """显式续问会复用原任务目标，但重新检索并生成本轮证据，不复用旧结果。"""
    frame = dict(task.task_frame or {})
    frame["supplements"] = _append_supplement(frame, question)
    task.task_frame = frame
    task.source_message_ids = _append_message_id(task.source_message_ids, source_message_id)
    task.status = TASK_READY
    task.completed_at = None
    return ConversationTaskTurn(task, _effective_question(task), _route_override(task))


def _record_task_interpretation(
    task: ConversationTask, source_message_id: str, decision: TaskRelationshipDecision
) -> None:
    """把关系判断留在任务帧中，便于审计而不把模型判断当成业务查询条件。"""
    frame = dict(task.task_frame or {})
    frame["last_interpretation"] = {
        "source_message_id": source_message_id,
        "relation": decision.relation,
        "changed_slots": list(decision.changed_slots),
        "confidence": decision.confidence,
        "reason": decision.reason,
        "source": decision.source,
    }
    task.task_frame = frame


def _waiting_turn(
    task: ConversationTask, temporal_resolution: TemporalResolution = TemporalResolution("no_time")
) -> ConversationTaskTurn:
    missing = "、".join(str(item) for item in task.missing_slots) or "必要条件"
    frame = dict(task.task_frame or {})
    time_clarification = frame.get("time_clarification")
    clarification = (
        str(time_clarification)
        if isinstance(time_clarification, str) and time_clarification
        else (
            f"要完成当前{_task_label(task.task_type)}，还需要确认：{missing}。"
            "请直接补充明确的日期范围，例如“本周是指 2026-09-09 至 2026-09-15”。"
        )
    )
    return ConversationTaskTurn(
        task,
        _effective_question(task),
        clarification=clarification,
        temporal_resolution=temporal_resolution,
    )


def _effective_question(task: ConversationTask) -> str:
    frame = dict(task.task_frame or {})
    base_question = str(frame.get("base_question") or "")
    supplements = [str(item) for item in frame.get("supplements", [])]
    parts = [base_question]
    periods = frame.get("resolved_periods") or frame.get("comparison_periods")
    if isinstance(periods, list):
        rendered_periods = "；".join(
            f"{item['label']} {item['start_date']} 至 {item['end_date']}"
            for item in periods
            if isinstance(item, dict)
            and isinstance(item.get("label"), str)
            and isinstance(item.get("start_date"), str)
            and isinstance(item.get("end_date"), str)
        )
        if rendered_periods:
            parts.append("已确认的受控时间范围：" + rendered_periods)
    if supplements:
        parts.append("用户后续补充条件：" + "；".join(supplements))
    return "\n\n".join(part for part in parts if part)


def _route_override(task: ConversationTask) -> RetrievalRoute | None:
    if task.route_mode not in {"metrics", "knowledge", "hybrid", "web", "web_hybrid"}:
        return None
    return RetrievalRoute(task.route_mode, None, 1.0, False, "conversation_task_continuation")


async def _latest_task(
    session: AsyncSession, conversation_id: str, statuses: tuple[str, ...]
) -> ConversationTask | None:
    return await session.scalar(
        select(ConversationTask)
        .where(
            ConversationTask.conversation_id == conversation_id,
            ConversationTask.status.in_(statuses),
        )
        .order_by(ConversationTask.updated_at.desc(), ConversationTask.created_at.desc())
        .limit(1)
    )


def _infer_task_type(question: str) -> str:
    lowered = question.lower()
    if _is_metric_comparison(lowered):
        return TASK_METRIC_COMPARISON
    if any(marker in lowered for marker in ("复盘", "归因", "效果评估")):
        return TASK_REVIEW
    if any(marker in lowered for marker in ("分析", "原因", "建议", "优化")) and (
        any(marker in lowered for marker in _METRIC_MARKERS)
        or any(marker in lowered for marker in _KNOWLEDGE_MARKERS)
    ):
        return TASK_HYBRID_ANALYSIS
    if any(marker in lowered for marker in _KNOWLEDGE_MARKERS):
        return TASK_KNOWLEDGE_QA
    return TASK_METRIC_QUERY


def _is_metric_comparison(lowered: str) -> bool:
    return (
        any(marker in lowered for marker in _METRIC_MARKERS)
        and any(marker in lowered for marker in _COMPARISON_MARKERS)
    )


def _append_supplement(frame: dict[str, object], question: str) -> list[str]:
    current = [str(item) for item in frame.get("supplements", [])]
    return [*current[-(_MAX_SUPPLEMENTS - 1) :], _truncate(question)]


def _append_message_id(message_ids: list[str] | None, source_message_id: str) -> list[str]:
    return list(dict.fromkeys([*(message_ids or []), source_message_id]))


def _temporal_task_context(task: ConversationTask | None) -> str | None:
    """仅向时间解析器提供继续任务所需的最小上下文，不传递历史回答。"""
    if task is None:
        return None
    frame = dict(task.task_frame or {})
    return f"任务类型={task.task_type}；原问题={str(frame.get('base_question') or '')[:600]}"


def _truncate(value: str) -> str:
    return value.strip()[:_FRAME_TEXT_LIMIT]


def _task_label(task_type: str) -> str:
    return {
        TASK_METRIC_QUERY: "指标查询",
        TASK_METRIC_COMPARISON: "指标对比",
        TASK_KNOWLEDGE_QA: "知识库问答",
        TASK_HYBRID_ANALYSIS: "综合分析",
        TASK_REVIEW: "活动复盘",
    }.get(task_type, "运营任务")
