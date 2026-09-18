from app.agent.task_orchestrator import MainAgentOrchestrator
from app.models import ConversationTask
from app.services.conversations.tasks import TASK_HYBRID_ANALYSIS, TASK_METRIC_COMPARISON


def test_causal_analysis_delegates_evidence_to_the_review_agent() -> None:
    task = ConversationTask(conversation_id="conversation-1", task_type=TASK_HYBRID_ANALYSIS)

    plan = MainAgentOrchestrator().plan(task)

    assert plan.route_mode == "hybrid"
    assert plan.use_review_agent is True
    assert [action.action_type for action in plan.actions] == [
        "confirm_constraints",
        "query_metrics",
        "query_knowledge",
        "evaluate_evidence",
        "review",
        "synthesize",
    ]


def test_plain_metric_comparison_remains_data_only() -> None:
    task = ConversationTask(conversation_id="conversation-1", task_type=TASK_METRIC_COMPARISON)

    plan = MainAgentOrchestrator().plan(task)

    assert plan.route_mode == "metrics"
    assert plan.use_review_agent is False
    assert [action.action_type for action in plan.actions] == [
        "confirm_constraints",
        "query_metrics",
        "evaluate_evidence",
        "synthesize",
    ]


def test_activity_review_with_metrics_and_materials_uses_both_evidence_sources() -> None:
    task = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_METRIC_COMPARISON,
        effective_constraints={
            "execution_intent": {
                "operation": "review",
                "evidence_sources": ["metrics", "knowledge"],
                "output_scope": "review",
                "allow_metric_drilldown": True,
            }
        },
    )

    plan = MainAgentOrchestrator().plan(task)

    assert plan.route_mode == "hybrid"
    assert plan.use_review_agent is True
    assert [action.action_type for action in plan.actions] == [
        "confirm_constraints",
        "query_metrics",
        "query_knowledge",
        "evaluate_evidence",
        "review",
        "synthesize",
    ]
