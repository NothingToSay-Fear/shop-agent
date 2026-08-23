from datetime import date
from decimal import Decimal

from app.seed import ACTIVITY_METRIC_SEEDS


def test_spring_review_fixture_matches_documented_result() -> None:
    """春季上新复盘中的活动 GMV 与支付订单应能被初始化数据准确复核。"""
    rows = [
        item
        for item in ACTIVITY_METRIC_SEEDS
        if date(2026, 3, 8) <= item["metric_date"] <= date(2026, 3, 14)
    ]

    assert sum(item["paid_order_count"] for item in rows) == 312
    assert sum((item["paid_gmv"] for item in rows), Decimal("0")) == Decimal("86400.00")


def test_qixi_fixture_contains_gift_box_and_combo_sales() -> None:
    """七夕方案中的礼盒与组合款均应有活动期经营记录。"""
    product_ids = {
        item["product_id"]
        for item in ACTIVITY_METRIC_SEEDS
        if date(2026, 8, 10) <= item["metric_date"] <= date(2026, 8, 22)
    }

    assert product_ids == {"demo-product-004", "demo-product-005"}


def test_618_fixture_covers_all_campaign_phases() -> None:
    """618 测试数据覆盖预热、正式和返场期，供规则与指标联合验证。"""
    rows = [
        item
        for item in ACTIVITY_METRIC_SEEDS
        if date(2026, 6, 1) <= item["metric_date"] <= date(2026, 6, 20)
    ]

    assert len(rows) == 40
    assert {item["channel"] for item in rows} == {"内容种草（618）", "搜索广告（618）"}
