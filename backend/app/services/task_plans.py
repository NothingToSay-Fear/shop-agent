"""持久化 Plan-and-Execute 动作、执行尝试与重规划历史。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.task_orchestrator import DelegationPlan, MainAgentOrchestrator, PlannedAction
from app.agent.tools.tracker import ToolCallAudit
from app.models import ConversationTask, TaskPlan, TaskPlanStep, TaskPlanStepAttempt
from app.services.conversation_tasks import TASK_WAITING_CLARIFICATION


@dataclass(frozen=True)
class ExecutablePlanAction:
    """从数据库读取、实际驱动执行器的不可变 Action 快照。"""

    step_id: str
    key: str
    title: str
    action_type: str
    tool_name: str | None
    depends_on: tuple[str, ...]
    action_input: dict[str, object]
    expected_output: str | None
    capability_requirement: dict[str, object]
    condition: dict[str, object]
    plan_version: int


@dataclass
class PlanExecutionController:
    """将执行器的状态变更写回同一份持久化计划。"""

    session: AsyncSession
    plan: TaskPlan

    async def begin(self, action: ExecutablePlanAction) -> TaskPlanStepAttempt:
        attempt = await start_action(self.session, action)
        await self.session.commit()
        return attempt

    async def finish(
        self,
        action: ExecutablePlanAction,
        attempt: TaskPlanStepAttempt,
        *,
        status: str,
        result_summary: str,
        evidence_references: tuple[str, ...] = (),
        error_message: str | None = None,
    ) -> None:
        await complete_action(
            self.session,
            action,
            attempt,
            status=status,
            result_summary=result_summary,
            evidence_references=evidence_references,
            error_message=error_message,
        )
        await self.session.commit()

    async def append_replan(
        self, actions: tuple[PlannedAction, ...], reason: str
    ) -> list[ExecutablePlanAction]:
        result = await append_replanned_actions(self.session, self.plan, actions, reason)
        await self.session.commit()
        return result


async def ensure_task_plan(
    session: AsyncSession,
    task: ConversationTask,
    delegation_plan: DelegationPlan | None = None,
) -> TaskPlan:
    """创建首版可执行计划；执行器只读取该持久化版本，不重建模板步骤。"""
    plan = await session.scalar(
        select(TaskPlan).where(TaskPlan.task_id == task.id, TaskPlan.revision == task.revision)
    )
    if plan is not None:
        await _upgrade_legacy_plan(session, plan, delegation_plan or MainAgentOrchestrator().plan(task))
        return plan

    planned = delegation_plan or MainAgentOrchestrator().plan(task)
    plan = TaskPlan(
        task_id=task.id,
        revision=task.revision,
        status="waiting" if task.status == TASK_WAITING_CLARIFICATION else "ready",
        summary=_plan_summary(planned),
        plan_version=1,
        budget=dict(planned.budget),
        planning_context={"kind": planned.kind, "route_mode": planned.route_mode},
    )
    session.add(plan)
    await session.flush()
    await _append_actions(session, plan, planned.actions, plan_version=plan.plan_version)
    return plan


async def load_executable_actions(session: AsyncSession, plan: TaskPlan) -> list[ExecutablePlanAction]:
    """加载尚未结束的动作；依赖和顺序均来自数据库而非代码分支。"""
    steps = list(
        await session.scalars(
            select(TaskPlanStep)
            .where(TaskPlanStep.plan_id == plan.id)
            .order_by(TaskPlanStep.ordinal, TaskPlanStep.created_at)
        )
    )
    return [_to_executable(step) for step in steps if step.status in {"pending", "running"}]


async def start_task_plan(session: AsyncSession, plan: TaskPlan) -> None:
    if plan.status != "waiting":
        plan.status = "executing"


async def start_action(
    session: AsyncSession, action: ExecutablePlanAction
) -> TaskPlanStepAttempt:
    step = await session.get(TaskPlanStep, action.step_id)
    if step is None:
        raise ValueError("计划动作不存在")
    count = await session.scalar(
        select(func.count(TaskPlanStepAttempt.id)).where(TaskPlanStepAttempt.step_id == step.id)
    )
    attempt = TaskPlanStepAttempt(
        step_id=step.id,
        attempt_number=int(count or 0) + 1,
        status="running",
        input_snapshot=dict(action.action_input),
    )
    step.status = "running"
    session.add(attempt)
    await session.flush()
    return attempt


async def complete_action(
    session: AsyncSession,
    action: ExecutablePlanAction,
    attempt: TaskPlanStepAttempt,
    *,
    status: str,
    result_summary: str,
    evidence_references: tuple[str, ...] = (),
    error_message: str | None = None,
) -> None:
    step = await session.get(TaskPlanStep, action.step_id)
    if step is None:
        raise ValueError("计划动作不存在")
    terminal = "completed" if status in {"success", "empty", "skipped", "completed"} else "failed"
    now = datetime.now(UTC)
    step.status = terminal
    step.result_summary = result_summary[:2000]
    step.evidence_references = list(evidence_references)
    step.error_message = error_message[:2000] if error_message else None
    step.completed_at = now
    attempt.status = terminal
    attempt.result_summary = result_summary[:2000]
    attempt.evidence_references = list(evidence_references)
    attempt.error_message = error_message[:2000] if error_message else None
    attempt.completed_at = now


async def append_replanned_actions(
    session: AsyncSession,
    plan: TaskPlan,
    actions: tuple[PlannedAction, ...],
    reason: str,
) -> list[ExecutablePlanAction]:
    """追加不可变的新版本动作，保留旧步骤和尝试供审计与恢复使用。"""
    if not actions:
        return []
    max_replans = int((plan.budget or {}).get("max_replans", 0))
    current_replans = max(0, (plan.plan_version or 1) - 1)
    if current_replans >= max_replans:
        return []
    current_action_count = await session.scalar(
        select(func.count(TaskPlanStep.id)).where(TaskPlanStep.plan_id == plan.id)
    )
    if int(current_action_count or 0) + len(actions) > int((plan.budget or {}).get("max_actions", 0)):
        return []
    plan.plan_version += 1
    context = dict(plan.planning_context or {})
    replan_reasons = list(context.get("replan_reasons", []))
    replan_reasons.append(reason[:300])
    context["replan_reasons"] = replan_reasons[-max_replans:]
    plan.planning_context = context
    await _append_actions(session, plan, actions, plan_version=plan.plan_version)
    await session.flush()
    return await load_executable_actions(session, plan)


async def complete_task_plan(
    session: AsyncSession, plan: TaskPlan, tool_calls: list[ToolCallAudit]
) -> None:
    """只有所有动作进入终态后才完成计划；工具轨迹已在每个动作中持久化。"""
    steps = list(await session.scalars(select(TaskPlanStep).where(TaskPlanStep.plan_id == plan.id)))
    if any(step.status in {"pending", "running"} for step in steps):
        plan.status = "blocked"
        return
    plan.status = "completed" if not any(step.status == "failed" for step in steps) else "failed"


async def fail_task_plan(session: AsyncSession, plan: TaskPlan, error_message: str) -> None:
    plan.status = "failed"
    steps = list(await session.scalars(select(TaskPlanStep).where(TaskPlanStep.plan_id == plan.id)))
    for step in steps:
        if step.status == "running":
            step.status = "failed"
            step.error_message = error_message[:500]
            step.completed_at = datetime.now(UTC)


async def _append_actions(
    session: AsyncSession,
    plan: TaskPlan,
    actions: tuple[PlannedAction, ...],
    *,
    plan_version: int,
) -> None:
    existing = list(
        await session.scalars(select(TaskPlanStep).where(TaskPlanStep.plan_id == plan.id))
    )
    ordinal = max((step.ordinal for step in existing), default=-1) + 1
    existing_keys = {step.step_key for step in existing}
    generated_keys: dict[str, str] = {}
    new_steps: list[TaskPlanStep] = []
    for action in actions:
        key = action.key if action.key not in existing_keys else f"{action.key}_v{plan_version}"
        existing_keys.add(key)
        generated_keys[action.key] = key
        depends_on = [
            generated_keys.get(
                dependency,
                dependency if dependency in existing_keys else f"{dependency}_v{plan_version - 1}",
            )
            for dependency in action.depends_on
        ]
        new_steps.append(
            TaskPlanStep(
                plan_id=plan.id,
                step_key=key,
                title=action.title,
                action_type=action.action_type,
                tool_name=action.tool_name,
                depends_on=depends_on,
                action_input=dict(action.action_input),
                expected_output=action.expected_output,
                capability_requirement=dict(action.capability_requirement),
                condition=dict(action.condition),
                plan_version=plan_version,
                ordinal=ordinal,
                status="pending",
            )
        )
        ordinal += 1
    session.add_all(new_steps)
    await session.flush()


async def _upgrade_legacy_plan(
    session: AsyncSession, plan: TaskPlan, planned: DelegationPlan
) -> None:
    """使升级前仍未结束的模板步骤安全过渡到 Action Plan。"""
    steps = list(await session.scalars(select(TaskPlanStep).where(TaskPlanStep.plan_id == plan.id)))
    legacy_open_steps = [
        step
        for step in steps
        if step.action_type == "tool" and step.status in {"pending", "running"}
    ]
    if not legacy_open_steps:
        return
    now = datetime.now(UTC)
    for step in legacy_open_steps:
        step.status = "skipped"
        step.result_summary = "该旧版模板步骤已迁移到新的持久化 Action Plan"
        step.completed_at = now
    plan.plan_version = max(plan.plan_version or 1, 1) + 1
    context = dict(plan.planning_context or {})
    context["legacy_migrated"] = True
    plan.planning_context = context
    await _append_actions(session, plan, planned.actions, plan_version=plan.plan_version)


def _to_executable(step: TaskPlanStep) -> ExecutablePlanAction:
    return ExecutablePlanAction(
        step_id=step.id,
        key=step.step_key,
        title=step.title,
        action_type=step.action_type,
        tool_name=step.tool_name,
        depends_on=tuple(step.depends_on or ()),
        action_input=dict(step.action_input or {}),
        expected_output=step.expected_output,
        capability_requirement=dict(step.capability_requirement or {}),
        condition=dict(step.condition or {}),
        plan_version=step.plan_version,
    )


def _plan_summary(plan: DelegationPlan) -> str:
    titles = "、".join(action.title for action in plan.actions[:3])
    return f"{plan.kind}：{titles}"[:300]
