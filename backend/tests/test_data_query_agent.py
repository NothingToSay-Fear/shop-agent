import pytest

from app.agent.data_query_agent import DATA_QUERY_AGENT_NAME, DataQueryAgent
from app.services.conversation_context import ConversationContextSnapshot
from app.services.data_query_planner import build_data_query_plan


def test_explicit_task_metrics_are_a_complete_execution_contract() -> None:
    plan = build_data_query_plan(
        {"metrics": ["paid_gmv", "conversion_rate", "refund_rate"]},
        "分析 GMV、支付转化率和退款率",
    )

    assert plan.metric_codes == ("paid_gmv", "conversion_rate", "refund_rate")
    assert plan.capability_notes == ()


def test_causal_gmv_question_expands_to_its_queryable_driver_metrics() -> None:
    plan = build_data_query_plan(
        {"metrics": ["paid_gmv"], "analysis_goal": "经营归因"},
        "为什么本周 GMV 比上周低",
        {
            "paid_gmv": (
                "paid_order_count",
                "visitor_count",
                "conversion_rate",
                "average_order_value",
            )
        },
    )

    assert plan.metric_codes == (
        "paid_gmv",
        "paid_order_count",
        "visitor_count",
        "conversion_rate",
        "average_order_value",
    )


def test_task_periods_override_activity_mentions_in_the_original_question() -> None:
    plan = build_data_query_plan(
        {
            "metrics": ["paid_gmv"],
            "periods": [
                {
                    "label": "2026-06-01 到 2026-06-18",
                    "start_date": "2026-06-01",
                    "end_date": "2026-06-18",
                }
            ],
        },
        "618 活动复盘",
    )

    assert plan.periods == (("2026-06-01 到 2026-06-18", "2026-06-01", "2026-06-18"),)


def test_product_refund_request_reports_missing_detail_source_without_dropping_aggregate_metrics() -> None:
    plan = build_data_query_plan(
        {"metrics": ["refund_rate"]},
        "优先分析高退款商品和对应的活动规则",
    )

    assert plan.metric_codes == ("refund_rate",)
    assert plan.dimensions == ("product",)
    assert plan.unsupported_requirements
    assert "daily_metrics" in plan.capability_notes[0]


@pytest.mark.asyncio
async def test_data_query_agent_executes_only_the_structured_metric_plan() -> None:
    class _MetricTool:
        payload: dict[str, object] | None = None

        async def ainvoke(self, payload: dict[str, object]) -> None:
            self.payload = payload

    tool = _MetricTool()
    agent = DataQueryAgent(
        {
            "metrics": ["paid_gmv", "conversion_rate", "refund_rate"],
            "periods": [
                {
                    "label": "618",
                    "start_date": "2026-06-01",
                    "end_date": "2026-06-18",
                }
            ],
        },
        "分析高退款商品",
    )

    plan = await agent.execute(
        tool,  # type: ignore[arg-type]
        "618 活动复盘",
        ConversationContextSnapshot(),
    )

    assert DATA_QUERY_AGENT_NAME == "data_query_agent"
    assert plan.metric_codes == ("paid_gmv", "conversion_rate", "refund_rate")
    assert tool.payload is not None
    assert tool.payload["metric_codes"] == ["paid_gmv", "conversion_rate", "refund_rate"]
    assert tool.payload["query_periods"] == [
        {"label": "618", "start_date": "2026-06-01", "end_date": "2026-06-18"}
    ]
