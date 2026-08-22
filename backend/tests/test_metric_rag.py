from types import SimpleNamespace

from app.services.metric_rag import (
    MetricDocument,
    _calculate_metric,
    _resolve_dependencies,
    retrieve_metrics,
)


def test_rag_retrieves_metric_by_business_alias() -> None:
    """运营人员使用“成交额”时，应能命中支付 GMV 指标而非全量查询。"""
    gmv = MetricDocument(
        metric_code="paid_gmv",
        name="支付 GMV",
        description="已支付订单的成交金额总和。",
        aliases=["成交额", "销售额"],
        embedding=[1.0, 0.0],
    )
    visitors = MetricDocument(
        metric_code="visitor_count",
        name="访客数",
        description="访问页面的去重访客数量。",
        aliases=["流量", "UV"],
        embedding=[0.0, 1.0],
    )

    result = retrieve_metrics("帮我看一下本周成交额", [gmv, visitors], [1.0, 0.0])

    assert result[0].metric_code == "paid_gmv"


def test_dependency_resolution_only_includes_required_metrics() -> None:
    """转化率只需要支付订单数和访客数两个前置指标。"""
    definitions = {
        "paid_order_count": SimpleNamespace(dependency_codes=[]),
        "visitor_count": SimpleNamespace(dependency_codes=[]),
        "conversion_rate": SimpleNamespace(
            dependency_codes=["paid_order_count", "visitor_count"]
        ),
    }

    result = _resolve_dependencies(["conversion_rate"], definitions)

    assert result == ["paid_order_count", "visitor_count", "conversion_rate"]


def test_registered_calculation_formula_uses_only_dependencies() -> None:
    """派生指标以已查询的前置值计算，不执行动态表达式。"""
    value = _calculate_metric(
        "paid_order_count / visitor_count", {"paid_order_count": 80, "visitor_count": 2000}
    )

    assert value == 0.04
