import pytest
from sqlalchemy import select

from app.agent.task_orchestrator import MainAgentOrchestrator
from app.database import SessionLocal
from app.models import Conversation, ConversationTask, TaskPlanStepAttempt, User
from app.services.task_plans import (
    append_replanned_actions,
    complete_action,
    ensure_task_plan,
    load_executable_actions,
    start_action,
)


pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_persisted_plan_actions_drive_attempt_audit() -> None:
    async with SessionLocal() as session:
        user = User(username="plan-execute-test", display_name="Plan Execute Test", password_hash="not-used")
        session.add(user)
        await session.flush()
        conversation = Conversation(title="Plan Execute Test", user_id=user.id)
        session.add(conversation)
        await session.flush()
        task = ConversationTask(
            conversation_id=conversation.id,
            task_type="metric_query",
            status="ready",
            revision=1,
            effective_constraints={"metrics": ["paid_gmv"]},
        )
        session.add(task)
        await session.flush()

        plan = await ensure_task_plan(session, task, MainAgentOrchestrator().plan(task))
        actions = await load_executable_actions(session, plan)
        confirm = actions[0]
        attempt = await start_action(session, confirm)
        await complete_action(
            session,
            confirm,
            attempt,
            status="completed",
            result_summary="已确认任务约束",
        )
        await session.flush()

        stored_attempt = await session.get(TaskPlanStepAttempt, attempt.id)
        assert [action.action_type for action in actions] == [
            "confirm_constraints",
            "query_metrics",
            "evaluate_evidence",
            "synthesize",
        ]
        assert stored_attempt is not None
        assert stored_attempt.status == "completed"
        assert stored_attempt.input_snapshot == {}
        followups = MainAgentOrchestrator().replan(
            MainAgentOrchestrator().plan(task),
            next_metric_codes=("visitor_count",),
            reason="需要继续验证驱动",
        )
        pending = await append_replanned_actions(session, plan, followups, "需要继续验证驱动")
        assert plan.plan_version == 2
        assert any(action.key.startswith("followup_metrics") for action in pending)
        followup_evaluation = next(
            action for action in pending if action.key.startswith("followup_evaluation")
        )
        assert followup_evaluation.depends_on == ("followup_metrics",)
        await session.rollback()
