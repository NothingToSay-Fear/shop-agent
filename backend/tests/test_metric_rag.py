from datetime import date
from types import SimpleNamespace

import pytest

from app.services.metric_rag import (
    MetricDocument,
    MetricQueryConstraints,
    MetricQueryPlanError,
    _calculate_metric,
    _format_unit_comparison,
    _parse_explicit_date_range,
    _resolve_period,
    _resolve_dependencies,
    build_metric_query_plan,
    retrieve_metrics,
)


class _LatestDateSession:
    """仅模拟默认周期解析所需的最新可用数据日期。"""

    def __init__(self, latest_date: date | None) -> None:
        self.latest_date = latest_date

    async def scalar(self, _statement: object) -> date | None:
        return self.latest_date


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


def test_explicit_date_range_uses_controlled_date_values() -> None:
    """活动日期可解析为受控日期参数，不能把原始问题拼入 SQL。"""
    period = _parse_explicit_date_range("查询 2026 年 6 月 6 日至 6 月 18 日的 618 GMV")

    assert period == (date(2026, 6, 6), date(2026, 6, 18))


@pytest.mark.asyncio
async def test_default_period_uses_latest_data_not_later_than_business_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """预置的未来演示数据不能被没有日期条件的问题默认查询。"""
    monkeypatch.setattr("app.services.metric_rag.current_business_date", lambda: date(2026, 9, 15))

    period = await _resolve_period(_LatestDateSession(date(2026, 9, 15)), "查看 GMV")

    assert period == (date(2026, 9, 9), date(2026, 9, 15))


def test_metric_query_plan_keeps_each_explicit_period_as_a_controlled_unit() -> None:
    """同一句中的多个明确日期范围应拆成多个查询单元，而非只保留第一个范围。"""
    plan = build_metric_query_plan(
        "对比 2026 年 6 月 1 日至 6 月 20 日和 2026 年 8 月 10 日至 8 月 22 日的 GMV"
    )

    assert [(item.start_date, item.end_date) for item in plan.units] == [
        (date(2026, 6, 1), date(2026, 6, 20)),
        (date(2026, 8, 10), date(2026, 8, 22)),
    ]


def test_metric_query_plan_uses_multiple_activities_before_inherited_single_period() -> None:
    """本轮明确比较多个活动时，不能被会话遗留的单一日期范围覆盖。"""
    plan = build_metric_query_plan(
        "对比 618 和七夕的 GMV、订单量",
        MetricQueryConstraints(date(2026, 6, 1), date(2026, 6, 20)),
    )

    assert [item.label for item in plan.units] == ["618", "七夕"]
    assert [(item.start_date, item.end_date) for item in plan.units] == [
        (date(2026, 6, 1), date(2026, 6, 20)),
        (date(2026, 8, 10), date(2026, 8, 22)),
    ]


def test_metric_query_plan_can_combine_an_activity_and_explicit_period() -> None:
    """活动期和手动日期区间同时出现时，也应形成完整的对比计划。"""
    plan = build_metric_query_plan("对比 618 与 2026 年 8 月 10 日至 8 月 22 日的 GMV")

    assert [item.label for item in plan.units] == ["618", "2026 年 8 月 10 日至 8 月 22 日"]


def test_metric_query_plan_rejects_excessive_periods_instead_of_partially_querying() -> None:
    """超出上限时必须整体拒绝，避免用户误以为得到的是完整比较结果。"""
    question = "；".join(
        f"2026 年 {month} 月 1 日至 {month} 月 2 日" for month in range(1, 6)
    )

    with pytest.raises(MetricQueryPlanError, match="最多支持 4 个"):
        build_metric_query_plan(question)


def test_multi_period_comparison_keeps_metric_units_explicit() -> None:
    """跨区间比较必须保留原始数值单位，比例指标则使用百分点。"""
    definition = SimpleNamespace(metric_code="paid_gmv", name="支付 GMV")
    comparison = _format_unit_comparison(
        definition,
        SimpleNamespace(label="618"),
        1000.0,
        SimpleNamespace(label="七夕"),
        1250.0,
    )

    assert comparison == "七夕 相比 618，支付 GMV 变化 250.00 元（+25.00%）。"
