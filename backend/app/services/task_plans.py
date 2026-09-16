"""为会话任务生成并维护受控、可审计的执行计划。"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.tracker import ToolCallAudit
from app.models import ConversationTask, TaskPlan, TaskPlanStep
from app.services.conversation_tasks import (
    TASK_HYBRID_ANALYSIS,
    TASK_METRIC_COMPARISON,
    TASK_REVIEW,
    TASK_WAITING_CLARIFICATION,
)


async def ensure_task_plan(session: AsyncSession, task: ConversationTask) -> TaskPlan:
    """每个任务版本只有一份计划；创建操作具备幂等性。"""
    plan = await session.scalar(
        select(TaskPlan).where(TaskPlan.task_id == task.id, TaskPlan.revision == task.revision)
    )
    if plan is not None:
        return plan

    plan = TaskPlan(
        task_id=task.id,
        revision=task.revision,
        status="waiting" if task.status == TASK_WAITING_CLARIFICATION else "ready",
        summary=_plan_summary(task),
    )
    session.add(plan)
    await session.flush()
    session.add_all(
        TaskPlanStep(
            plan_id=plan.id,
            step_key=step_key,
            title=title,
            tool_name=tool_name,
            depends_on=depends_on,
            status="pending",
        )
        for step_key, title, tool_name, depends_on in _plan_steps(task)
    )
    return plan


async def start_task_plan(session: AsyncSession, plan: TaskPlan) -> None:
    if plan.status == "waiting":
        return
    plan.status = "running"
    steps = list(await session.scalars(select(TaskPlanStep).where(TaskPlanStep.plan_id == plan.id)))
    for step in steps:
        if step.step_key == "confirm_constraints":
            step.status = "completed"
            step.result_summary = "已使用任务有效约束生成本轮受控执行范围。"
            step.completed_at = datetime.now(UTC)
        elif step.tool_name:
            step.status = "running"


async def complete_task_plan(
    session: AsyncSession, plan: TaskPlan, tool_calls: list[ToolCallAudit]
) -> None:
    steps = list(await session.scalars(select(TaskPlanStep).where(TaskPlanStep.plan_id == plan.id)))
    calls_by_tool = {call.tool_name: call for call in tool_calls}
    for step in steps:
        if step.step_key == "retrieve":
            step.status = "completed"
            step.result_summary = "已按实际路由完成本轮受控检索。"
            step.completed_at = datetime.now(UTC)
            continue
        if not step.tool_name:
            continue
        call = calls_by_tool.get(step.tool_name)
        if call is None:
            step.status = "skipped"
            step.result_summary = "本轮计划未要求该工具。"
        elif call.status in {"success", "empty", "skipped"}:
            step.status = "completed"
            step.result_summary = call.result_summary
            step.evidence_references = list(call.reference_ids)
        else:
            step.status = "failed"
            step.error_message = call.error_code or "工具调用失败"
        step.completed_at = datetime.now(UTC)
    for step in steps:
        if step.step_key == "synthesize":
            step.status = "completed"
            step.result_summary = "已基于完成步骤的受控证据生成回答。"
            step.completed_at = datetime.now(UTC)
    plan.status = "completed"


async def fail_task_plan(session: AsyncSession, plan: TaskPlan, error_message: str) -> None:
    plan.status = "failed"
    steps = list(await session.scalars(select(TaskPlanStep).where(TaskPlanStep.plan_id == plan.id)))
    for step in steps:
        if step.status == "running":
            step.status = "failed"
            step.error_message = error_message[:500]
            step.completed_at = datetime.now(UTC)


def _plan_summary(task: ConversationTask) -> str:
    return {
        TASK_REVIEW: "确认复盘口径、检索指标与资料、形成受控结论",
        TASK_HYBRID_ANALYSIS: "确认分析约束、检索指标与资料、形成受控结论",
        TASK_METRIC_COMPARISON: "确认比较区间、查询指标、形成比较结论",
    }.get(task.task_type, "确认任务约束、执行受控检索、生成回答")


def _plan_steps(task: ConversationTask) -> tuple[tuple[str, str, str | None, list[str]], ...]:
    base = [("confirm_constraints", "确认任务约束", None, [])]
    if task.task_type in {TASK_REVIEW, TASK_HYBRID_ANALYSIS}:
        base.extend(
            [
                ("query_metrics", "查询经营指标", "query_metric_rag", ["confirm_constraints"]),
                ("query_knowledge", "检索活动资料", "query_knowledge_rag", ["confirm_constraints"]),
            ]
        )
    elif task.task_type == TASK_METRIC_COMPARISON:
        base.append(("query_metrics", "查询对比指标", "query_metric_rag", ["confirm_constraints"]))
    else:
        base.append(("retrieve", "执行受控检索", None, ["confirm_constraints"]))
    base.append(("synthesize", "形成结论与待验证项", None, [item[0] for item in base[1:]]))
    return tuple(base)
