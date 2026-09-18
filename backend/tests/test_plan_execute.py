from app.agent.data_query_agent import DataQueryAgent
from app.agent.task_orchestrator import MainAgentOrchestrator
from app.agent.tools import AgentToolTracker
from app.services.analytics.metric_rag import MetricQueryContext


def test_data_query_agent_proposes_only_unqueried_registered_drivers() -> None:
    agent = DataQueryAgent(
        {"metrics": ["paid_gmv"]},
        "为什么 GMV 下降",
        driver_graph={"paid_gmv": ("paid_order_count",), "paid_order_count": ("visitor_count",)},
    )
    tracker = AgentToolTracker(
        metric_context=MetricQueryContext("指标结果", ("paid_gmv", "paid_order_count"))
    )

    observation = agent.observe(tracker)

    assert observation.sufficient is False
    assert observation.next_metric_codes == ("visitor_count",)


def test_main_agent_replan_returns_constrained_metric_actions() -> None:
    task = type("Task", (), {"task_type": "hybrid_analysis"})()
    plan = MainAgentOrchestrator().plan(task)

    actions = MainAgentOrchestrator().replan(
        plan, next_metric_codes=("visitor_count",), reason="需要验证下一层驱动"
    )

    assert [action.action_type for action in actions] == ["query_metrics", "evaluate_evidence"]
    assert actions[0].tool_name == "query_metric_rag"
    assert actions[0].action_input["metric_codes"] == ["visitor_count"]
