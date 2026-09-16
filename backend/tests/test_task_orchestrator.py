from app.agent.task_orchestrator import MainAgentOrchestrator
from app.models import ConversationTask
from app.services.conversation_tasks import TASK_HYBRID_ANALYSIS, TASK_METRIC_COMPARISON


def test_causal_analysis_delegates_evidence_to_the_review_agent() -> None:
    task = ConversationTask(conversation_id="conversation-1", task_type=TASK_HYBRID_ANALYSIS)

    plan = MainAgentOrchestrator().plan(task)

    assert plan.route_mode == "hybrid"
    assert plan.use_review_agent is True
    assert plan.batches == (("data_query", "knowledge_retrieval"), ("review",))


def test_plain_metric_comparison_remains_data_only() -> None:
    task = ConversationTask(conversation_id="conversation-1", task_type=TASK_METRIC_COMPARISON)

    plan = MainAgentOrchestrator().plan(task)

    assert plan.route_mode == "metrics"
    assert plan.use_review_agent is False
