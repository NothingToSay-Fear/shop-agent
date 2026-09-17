"""管理可跨轮延续的会话任务及其结构化有效约束。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ConversationTask, ConversationTaskEvent
from app.services.activity_periods import resolve_activity_periods
from app.services.data_query_planner import build_data_query_plan
from app.services.date_ranges import parse_explicit_date_ranges
from app.services.metric_analysis_graph import load_metric_analysis_driver_graph
from app.services.intent_router import RetrievalRoute
from app.services.task_interpreter import TaskRelationshipDecision, interpret_task_relationship
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

_METRIC_MARKERS = (
    ("paid_gmv", ("gmv", "成交额", "销售额")),
    ("conversion_rate", ("转化率",)),
    ("refund_rate", ("退款率",)),
    ("paid_order_count", ("订单",)),
    ("visitor_count", ("访客", "uv")),
    ("average_order_value", ("客单价",)),
)
_COMPARISON_MARKERS = ("比较", "对比", "相比", "环比", "同比", "差异")
_KNOWLEDGE_MARKERS = ("资料", "文档", "规则", "玩法", "手册", "复盘", "文件")
_CAUSAL_MARKERS = ("为什么", "为何", "原因", "归因", "下滑原因", "下降原因")
_FRAME_TEXT_LIMIT = 1200
_MAX_SUPPLEMENTS = 3
_INHERITANCE_MARKERS = ("保留", "沿用", "刚才", "上面", "上述", "原来的", "同一范围")
_TIME_OVERRIDE_MARKERS = ("改为", "换成", "切换为", "调整为")
_RELATIVE_TIME_MARKERS = ("本周", "这周", "上周", "本月", "这个月", "上月", "上个月", "本季度", "上季度")


@dataclass(frozen=True)
class ConversationTaskTurn:
    """本轮要执行的任务快照，或需要用户补充的最小问题。"""

    task: ConversationTask
    effective_question: str
    route_override: RetrievalRoute | None = None
    clarification: str | None = None
    temporal_resolution: TemporalResolution = TemporalResolution("no_time")

    @property
    def requires_clarification(self) -> bool:
        return self.clarification is not None


@dataclass(frozen=True)
class TaskConstraintAudit:
    """任务执行中实际使用的唯一约束快照。

它不复用 ``ConversationContext``：后者仅为旧会话条件提供兼容存储，不参与任务型问题的
检索、工具入参或回答生成。
    """

    summary: str
    actions: tuple[str, ...]
    snapshot: dict[str, object]


async def prepare_conversation_task(
    session: AsyncSession,
    conversation_id: str,
    source_message_id: str,
    question: str,
) -> ConversationTaskTurn:
    """将本轮输入合并进任务的有效状态，而不是重建文本拼接任务。"""
    pending = await _latest_task(session, conversation_id, (TASK_WAITING_CLARIFICATION, TASK_READY))
    completed = await _latest_task(session, conversation_id, (TASK_COMPLETED,))
    active_task = pending or completed
    temporal_resolution = await resolve_temporal_intent(
        session,
        question,
        task_context=_temporal_task_context(active_task),
    )

    if active_task is None:
        return await _create_new_task(
            session, conversation_id, source_message_id, question, temporal_resolution
        )

    decision = await interpret_task_relationship(active_task, question)
    if _requires_constraint_inheritance(question):
        # 分析目标可以升级，但明确沿用范围时不能将同一任务替换掉。
        decision = TaskRelationshipDecision(
            "revise",
            tuple(slot for slot in decision.changed_slots if slot != "time_range"),
            1.0,
            "用户明确要求继承已确认范围，仅更新本轮分析要求",
            "fallback",
        )
        temporal_resolution = TemporalResolution("no_time", source="local_gate")
    _record_task_interpretation(active_task, source_message_id, decision)

    if decision.relation == "cancel":
        active_task.status = TASK_CANCELLED
        active_task.completed_at = datetime.now(UTC)
        _record_task_event(session, active_task, "cancelled", source_message_id, decision)
        return ConversationTaskTurn(
            active_task,
            _effective_question(active_task),
            clarification="当前任务已取消。",
            temporal_resolution=TemporalResolution("no_time", source="local_gate"),
        )

    if decision.relation == "replace":
        active_task.status = TASK_SUPERSEDED
        active_task.completed_at = datetime.now(UTC)
        _record_task_event(session, active_task, "superseded", source_message_id, decision)
        replacement_input = decision.next_task_input or question
        replacement_temporal = await resolve_temporal_intent(
            session,
            replacement_input,
            task_context=None,
        )
        return await _create_new_task(
            session, conversation_id, source_message_id, replacement_input, replacement_temporal
        )

    return await _merge_into_existing_task(
        active_task, source_message_id, question, temporal_resolution, decision, session
    )


async def complete_conversation_task(
    task: ConversationTask, route: RetrievalRoute | None
) -> None:
    """只在本轮 Agent 正常完成后关闭任务；计划状态由调用方同步。"""
    task.status = TASK_COMPLETED
    task.completed_at = datetime.now(UTC)
    if route is not None:
        task.route_mode = route.mode


async def cancel_conversation_tasks(session: AsyncSession, conversation_id: str) -> None:
    """重置会话任务时取消可被后续输入继承的任务。"""
    tasks = list(
        await session.scalars(
            select(ConversationTask).where(
                ConversationTask.conversation_id == conversation_id,
                ConversationTask.status.in_(
                    (TASK_WAITING_CLARIFICATION, TASK_READY, TASK_COMPLETED)
                ),
            )
        )
    )
    for task in tasks:
        task.status = TASK_CANCELLED
        task.completed_at = datetime.now(UTC)
        _record_task_event(session, task, "cancelled", None, None)


def task_constraint_audit(task: ConversationTask) -> TaskConstraintAudit:
    """将任务有效约束作为运行审计快照。

任务型请求只从这个快照派生检索上下文，避免与旧的会话条件记录同时生效。
    """
    constraints = _constraints_from_task(task)
    parts: list[str] = []
    periods = constraints.get("periods")
    if isinstance(periods, list):
        rendered_periods = [
            f"{item['label']} {item['start_date']} 至 {item['end_date']}"
            for item in periods
            if isinstance(item, dict)
            and all(isinstance(item.get(key), str) for key in ("label", "start_date", "end_date"))
        ]
        if rendered_periods:
            parts.append("时间=" + "；".join(rendered_periods))
    metrics = constraints.get("metrics")
    if isinstance(metrics, list) and metrics:
        parts.append("指标=" + "、".join(str(item) for item in metrics if isinstance(item, str)))
    analysis_goal = constraints.get("analysis_goal")
    if isinstance(analysis_goal, str) and analysis_goal:
        parts.append("目标=" + analysis_goal)
    activities = constraints.get("activities")
    if isinstance(activities, list) and activities:
        parts.append("活动=" + "、".join(str(item) for item in activities if isinstance(item, str)))

    display = "；".join(parts) or "无显式约束"
    return TaskConstraintAudit(
        summary=f"本轮执行任务有效约束：{display}",
        actions=("本轮检索及生成仅使用会话任务的有效约束",),
        snapshot={
            "task_id": task.id,
            "task_revision": task.revision,
            "effective_constraints": constraints,
        },
    )


async def _create_new_task(
    session: AsyncSession,
    conversation_id: str,
    source_message_id: str,
    question: str,
    temporal_resolution: TemporalResolution,
) -> ConversationTaskTurn:
    task_type = _infer_task_type(question)
    constraints = _merge_constraints({}, question, temporal_resolution, task_type)
    constraints = await _attach_data_query_plan(session, task_type, constraints, question)
    pending_questions = _pending_questions(constraints, temporal_resolution)
    task = ConversationTask(
        conversation_id=conversation_id,
        task_type=task_type,
        status=TASK_WAITING_CLARIFICATION if pending_questions else TASK_READY,
        route_mode=_route_mode_for_constraints(constraints),
        revision=1,
        task_frame={
            "base_question": _truncate(question),
            "supplements": [],
            "temporal_resolution": _temporal_audit(temporal_resolution),
            "resolved_periods": list(constraints.get("periods", [])),
        },
        effective_constraints=constraints,
        pending_questions=pending_questions,
        missing_slots=["时间范围或数据类型"] if pending_questions else [],
        source_message_ids=[source_message_id],
    )
    session.add(task)
    await session.flush()
    _record_task_event(session, task, "created", source_message_id, None)
    return _turn_from_task(task, temporal_resolution)


async def _merge_into_existing_task(
    task: ConversationTask,
    source_message_id: str,
    question: str,
    temporal_resolution: TemporalResolution,
    decision: TaskRelationshipDecision,
    session: AsyncSession,
) -> ConversationTaskTurn:
    """continue/revise 只更新明确字段，未提及的已确认约束永远保留。"""
    frame = dict(task.task_frame or {})
    task.task_type = _refine_task_type(task.task_type, question)
    if task.task_type == TASK_HYBRID_ANALYSIS:
        # 因果追问不能继续沿用上一轮纯指标对比的 metrics 路由。
        task.route_mode = "hybrid"
    constraints = _merge_constraints(
        _constraints_from_task(task), question, temporal_resolution, task.task_type
    )
    constraints = await _attach_data_query_plan(session, task.task_type, constraints, question)
    pending_questions = _pending_questions(constraints, temporal_resolution)

    task.revision = max(task.revision or 1, 1) + 1
    task.effective_constraints = constraints
    task.route_mode = _route_mode_for_constraints(constraints)
    task.pending_questions = pending_questions
    task.missing_slots = ["时间范围或数据类型"] if pending_questions else []
    task.source_message_ids = _append_message_id(task.source_message_ids, source_message_id)
    task.status = TASK_WAITING_CLARIFICATION if pending_questions else TASK_READY
    task.completed_at = None
    frame["supplements"] = _append_supplement(frame, question)
    frame["resolved_periods"] = list(constraints.get("periods", []))
    frame["temporal_resolution"] = _temporal_audit(temporal_resolution)
    frame.pop("time_clarification", None)
    if pending_questions:
        frame["time_clarification"] = pending_questions[0]
    task.task_frame = frame
    _record_task_event(session, task, decision.relation, source_message_id, decision)
    return _turn_from_task(task, temporal_resolution)


def _continue_completed_task(
    task: ConversationTask, source_message_id: str, question: str
) -> ConversationTaskTurn:
    """兼容内部调用：在同一任务版本上恢复已完成任务，不创建新任务。"""
    frame = dict(task.task_frame or {})
    task.revision = max(task.revision or 1, 1) + 1
    task.source_message_ids = _append_message_id(task.source_message_ids, source_message_id)
    frame["supplements"] = _append_supplement(frame, question)
    task.task_frame = frame
    task.status = TASK_READY
    task.completed_at = None
    return _turn_from_task(task, TemporalResolution("no_time", source="local_gate"))


def _turn_from_task(task: ConversationTask, temporal_resolution: TemporalResolution) -> ConversationTaskTurn:
    if task.pending_questions:
        return ConversationTaskTurn(
            task,
            _effective_question(task),
            clarification=str(task.pending_questions[0]),
            temporal_resolution=temporal_resolution,
        )
    return ConversationTaskTurn(
        task,
        _effective_question(task),
        _route_override(task),
        temporal_resolution=temporal_resolution,
    )


def _merge_constraints(
    existing: dict[str, object],
    question: str,
    temporal_resolution: TemporalResolution,
    task_type: str,
) -> dict[str, object]:
    """执行 KEEP / REPLACE / ASK：只有 resolved 覆盖时间；clarify 不破坏旧值。"""
    constraints = dict(existing)
    if temporal_resolution.status == "resolved":
        constraints["periods"] = _serialize_periods(temporal_resolution)
        constraints["activities"] = [item[0] for item in resolve_activity_periods(question)]
    metrics = _extract_metrics(question)
    if metrics:
        constraints["metrics"] = metrics
    analysis_goal = _extract_analysis_goal(question)
    if analysis_goal:
        constraints["analysis_goal"] = analysis_goal
    constraints["execution_intent"] = _execution_intent(task_type, constraints, question)
    return constraints


def _execution_intent(
    task_type: str, constraints: dict[str, object], question: str
) -> dict[str, object]:
    """将输出目标、证据来源和下钻授权分别保存，避免单一 task_type 覆盖混合意图。"""
    lowered = question.lower()
    analysis_goal = constraints.get("analysis_goal")
    is_review = task_type == TASK_REVIEW or analysis_goal == "活动复盘"
    is_diagnose = task_type == TASK_HYBRID_ANALYSIS or analysis_goal == "经营归因"
    has_knowledge_request = any(marker in lowered for marker in _KNOWLEDGE_MARKERS)
    needs_metrics = task_type != TASK_KNOWLEDGE_QA
    needs_knowledge = has_knowledge_request or is_review or task_type == TASK_HYBRID_ANALYSIS

    if is_review:
        operation = "review"
        output_scope = "review"
    elif is_diagnose:
        operation = "causal_analysis"
        output_scope = "diagnose"
    elif task_type == TASK_METRIC_COMPARISON:
        operation = "comparison"
        output_scope = "comparison_only"
    else:
        operation = "lookup"
        output_scope = "lookup_only"

    sources: list[str] = []
    if needs_metrics:
        sources.append("metrics")
    if needs_knowledge:
        sources.append("knowledge")
    return {
        "operation": operation,
        "evidence_sources": sources,
        "output_scope": output_scope,
        "allow_metric_drilldown": output_scope in {"diagnose", "review"},
    }


def _route_mode_for_constraints(constraints: dict[str, object]) -> str | None:
    intent = constraints.get("execution_intent")
    sources = intent.get("evidence_sources") if isinstance(intent, dict) else []
    if not isinstance(sources, list):
        return None
    has_metrics = "metrics" in sources
    has_knowledge = "knowledge" in sources
    if has_metrics and has_knowledge:
        return "hybrid"
    if has_metrics:
        return "metrics"
    if has_knowledge:
        return "knowledge"
    return None


def _requires_constraint_inheritance(question: str) -> bool:
    """明确沿用旧条件时，以字段继承优先于关系模型的 replace 判断。"""
    lowered = question.lower()
    if "取消" in question or "停止" in question:
        return False
    if not any(marker in question for marker in _INHERITANCE_MARKERS):
        return False
    if parse_explicit_date_ranges(question):
        return False
    has_relative_time_override = any(marker in lowered for marker in _RELATIVE_TIME_MARKERS) and any(
        marker in question for marker in _TIME_OVERRIDE_MARKERS
    )
    return not has_relative_time_override


async def _attach_data_query_plan(
    session: AsyncSession, task_type: str, constraints: dict[str, object], question: str
) -> dict[str, object]:
    """把数据子 Agent 的受控计划随任务版本持久化，供执行与审计共用。"""
    planned_constraints = dict(constraints)
    planned_constraints.pop("data_query_plan", None)
    driver_graph = await load_metric_analysis_driver_graph(session)
    plan = build_data_query_plan(planned_constraints, question, driver_graph)
    planned_constraints["data_query_plan"] = plan.as_dict()
    return planned_constraints


def _constraints_from_task(task: ConversationTask) -> dict[str, object]:
    if task.effective_constraints:
        return dict(task.effective_constraints)
    frame = dict(task.task_frame or {})
    legacy_periods = frame.get("resolved_periods") or frame.get("comparison_periods") or []
    return {"periods": list(legacy_periods) if isinstance(legacy_periods, list) else []}


def _pending_questions(
    constraints: dict[str, object], temporal_resolution: TemporalResolution
) -> list[str]:
    periods = constraints.get("periods")
    if isinstance(periods, list) and periods:
        return []
    if temporal_resolution.status == "clarify" and temporal_resolution.clarification:
        return [temporal_resolution.clarification]
    return []


def _serialize_periods(temporal_resolution: TemporalResolution) -> list[dict[str, str]]:
    return [
        {"label": label, "start_date": start.isoformat(), "end_date": end.isoformat()}
        for label, start, end in temporal_resolution.periods
    ]


def _extract_metrics(question: str) -> list[str]:
    lowered = question.lower()
    return [code for code, aliases in _METRIC_MARKERS if any(alias.lower() in lowered for alias in aliases)]


def _extract_analysis_goal(question: str) -> str | None:
    if any(marker in question for marker in ("退款", "退货")):
        return "退款与售后分析"
    if "复盘" in question:
        return "活动复盘"
    if any(marker in question for marker in _CAUSAL_MARKERS):
        return "经营归因"
    if any(marker in question for marker in ("优化", "建议")):
        return "优化建议"
    return None


def _record_task_interpretation(
    task: ConversationTask, source_message_id: str, decision: TaskRelationshipDecision
) -> None:
    frame = dict(task.task_frame or {})
    frame["last_interpretation"] = {
        "source_message_id": source_message_id,
        "relation": decision.relation,
        "changed_slots": list(decision.changed_slots),
        "confidence": decision.confidence,
        "reason": decision.reason,
        "source": decision.source,
        "uses_next_task_input": decision.next_task_input is not None,
    }
    task.task_frame = frame


def _record_task_event(
    session: AsyncSession,
    task: ConversationTask,
    event_type: str,
    source_message_id: str | None,
    decision: TaskRelationshipDecision | None,
) -> None:
    session.add(
        ConversationTaskEvent(
            task_id=task.id,
            revision=task.revision or 1,
            event_type=event_type,
            source_message_id=source_message_id,
            details={
                "relation": decision.relation if decision else None,
                "changed_slots": list(decision.changed_slots) if decision else [],
                "constraints": dict(task.effective_constraints or {}),
            },
        )
    )


def _effective_question(task: ConversationTask) -> str:
    frame = dict(task.task_frame or {})
    base_question = str(frame.get("base_question") or "")
    supplements = [str(item) for item in frame.get("supplements", [])]
    constraints = _constraints_from_task(task)
    periods = constraints.get("periods", [])
    parts = [base_question]
    if isinstance(periods, list):
        rendered = "；".join(
            f"{item['label']} {item['start_date']} 至 {item['end_date']}"
            for item in periods
            if isinstance(item, dict)
            and all(isinstance(item.get(key), str) for key in ("label", "start_date", "end_date"))
        )
        if rendered:
            parts.append("已确认的受控时间范围：" + rendered)
    metrics = constraints.get("metrics")
    if isinstance(metrics, list) and metrics:
        parts.append("已确认的受控指标：" + "、".join(
            str(item) for item in metrics if isinstance(item, str)
        ))
    analysis_goal = constraints.get("analysis_goal")
    if isinstance(analysis_goal, str) and analysis_goal:
        parts.append("已确认的分析目标：" + analysis_goal)
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
    has_metric = any(alias in lowered for _, aliases in _METRIC_MARKERS for alias in aliases)
    # “为什么 GMV 比上周低”既含比较，也要求解释变化。归因意图优先于纯对比。
    if any(marker in lowered for marker in _CAUSAL_MARKERS) and has_metric:
        return TASK_HYBRID_ANALYSIS
    if any(alias in lowered for _, aliases in _METRIC_MARKERS for alias in aliases) and any(
        marker in lowered for marker in _COMPARISON_MARKERS
    ):
        return TASK_METRIC_COMPARISON
    if any(marker in lowered for marker in ("复盘", "归因", "效果评估")):
        return TASK_REVIEW
    if any(marker in lowered for marker in ("分析", "建议", "优化")) and (
        has_metric
        or any(marker in lowered for marker in _KNOWLEDGE_MARKERS)
    ):
        return TASK_HYBRID_ANALYSIS
    if any(marker in lowered for marker in _KNOWLEDGE_MARKERS):
        return TASK_KNOWLEDGE_QA
    return TASK_METRIC_QUERY


def _refine_task_type(current_task_type: str, question: str) -> str:
    """允许同一会话中的“为什么”追问将查询/对比任务升级为归因任务。"""
    inferred_task_type = _infer_task_type(question)
    if current_task_type == TASK_REVIEW or inferred_task_type == TASK_REVIEW:
        return TASK_REVIEW
    if inferred_task_type == TASK_HYBRID_ANALYSIS or (
        current_task_type in {TASK_METRIC_QUERY, TASK_METRIC_COMPARISON}
        and any(marker in question.lower() for marker in _CAUSAL_MARKERS)
    ):
        return TASK_HYBRID_ANALYSIS
    return current_task_type


def _append_supplement(frame: dict[str, object], question: str) -> list[str]:
    current = [str(item) for item in frame.get("supplements", [])]
    return [*current[-(_MAX_SUPPLEMENTS - 1) :], _truncate(question)]


def _append_message_id(message_ids: list[str] | None, source_message_id: str) -> list[str]:
    return list(dict.fromkeys([*(message_ids or []), source_message_id]))


def _temporal_task_context(task: ConversationTask | None) -> str | None:
    if task is None:
        return None
    constraints = _constraints_from_task(task)
    return (
        f"任务类型={task.task_type}；已确认约束={constraints}；"
        f"原问题={str(dict(task.task_frame or {}).get('base_question') or '')[:600]}"
    )


def _temporal_audit(resolution: TemporalResolution) -> dict[str, object]:
    return {"status": resolution.status, "source": resolution.source}


def _truncate(value: str) -> str:
    return value.strip()[:_FRAME_TEXT_LIMIT]
