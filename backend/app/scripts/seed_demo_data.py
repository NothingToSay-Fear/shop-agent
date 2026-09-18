"""初始化可重复执行的本地模拟经营数据。"""

import argparse
import asyncio
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import delete, select

from app.database import SessionLocal, create_tables
from app.models import DailyMetric, Product
from app.services.analytics.metric_rag import seed_metric_definitions

DEMO_PRODUCTS = (
    {
        "id": "demo-product-001",
        "sku": "DEMO-BAG-001",
        "name": "轻量通勤双肩包",
        "category": "箱包",
        "price": Decimal("299.00"),
        "highlights": "16 寸电脑收纳、防泼水面料、轻量通勤设计",
    },
    {
        "id": "demo-product-002",
        "sku": "DEMO-CUP-001",
        "name": "316 不锈钢保温杯",
        "category": "居家日用",
        "price": Decimal("159.00"),
        "highlights": "长效保温、一键开盖、适合通勤与运动",
    },
    {
        "id": "demo-product-003",
        "sku": "DEMO-LAMP-001",
        "name": "护眼学习台灯",
        "category": "家居电器",
        "price": Decimal("239.00"),
        "highlights": "无可视频闪、多档调光、定时休息提醒",
    },
    {
        "id": "demo-product-004",
        "sku": "DEMO-GIFT-BOX-001",
        "name": "保温杯礼盒",
        "category": "礼赠",
        "price": Decimal("239.00"),
        "highlights": "316 不锈钢保温杯、礼品袋与祝福卡，适合职场礼赠",
    },
    {
        "id": "demo-product-005",
        "sku": "DEMO-GIFT-COMBO-001",
        "name": "通勤双肩包保温杯礼盒组合",
        "category": "礼赠",
        "price": Decimal("399.00"),
        "highlights": "通勤双肩包与保温杯礼盒组合，适合情侣与职场送礼场景",
    },
)

DEMO_YEAR = 2026
DEMO_CHANNELS = (("内容种草", Decimal("1.00")), ("搜索广告", Decimal("0.72")))


def _build_yearly_base_metric_seeds() -> tuple[dict[str, object], ...]:
    """构造全年连续的基础模拟数据，数值固定以保证本地测试可复现。"""
    rows: list[dict[str, object]] = []
    month_factors = {
        1: Decimal("0.82"),
        2: Decimal("0.76"),
        3: Decimal("0.98"),
        4: Decimal("1.02"),
        5: Decimal("1.08"),
        6: Decimal("1.24"),
        7: Decimal("0.94"),
        8: Decimal("1.10"),
        9: Decimal("0.97"),
        10: Decimal("1.05"),
        11: Decimal("1.32"),
        12: Decimal("1.18"),
    }
    start_date = date(DEMO_YEAR, 1, 1)
    for offset in range(365):
        metric_date = start_date + timedelta(days=offset)
        weekend_factor = Decimal("1.12") if metric_date.weekday() >= 5 else Decimal("1.00")
        campaign_factor = Decimal("1.00")
        price_factor = Decimal("1.00")
        if date(DEMO_YEAR, 6, 1) <= metric_date <= date(DEMO_YEAR, 6, 20):
            campaign_factor, price_factor = Decimal("1.28"), Decimal("0.94")
        elif date(DEMO_YEAR, 8, 10) <= metric_date <= date(DEMO_YEAR, 8, 22):
            campaign_factor, price_factor = Decimal("1.16"), Decimal("0.97")
        elif date(DEMO_YEAR, 3, 8) <= metric_date <= date(DEMO_YEAR, 3, 14):
            campaign_factor = Decimal("1.14")
        for product_index, product in enumerate(DEMO_PRODUCTS):
            for channel_index, (channel, channel_factor) in enumerate(DEMO_CHANNELS):
                weekday_wave = Decimal("1.00") + Decimal((offset % 7) - 3) * Decimal("0.015")
                base_visitors = Decimal(460 + product_index * 78 + (offset % 11) * 13)
                visitors = int(
                    base_visitors
                    * month_factors[metric_date.month]
                    * weekend_factor
                    * campaign_factor
                    * channel_factor
                    * weekday_wave
                )
                conversion = Decimal("0.031") + Decimal(product_index) * Decimal("0.003")
                if channel_index == 0:
                    conversion += Decimal("0.004")
                if campaign_factor > Decimal("1.00"):
                    conversion += Decimal("0.003")
                paid_orders = max(1, round(Decimal(visitors) * conversion))
                refund_orders = (
                    1 if (offset + product_index * 3 + channel_index * 5) % 17 == 0 else 0
                )
                rows.append(
                    {
                        "metric_date": metric_date,
                        "channel": channel,
                        "product_id": product["id"],
                        "visitor_count": visitors,
                        "paid_order_count": paid_orders,
                        "paid_gmv": (Decimal(paid_orders) * product["price"] * price_factor).quantize(
                            Decimal("0.01")
                        ),
                        "refund_order_count": refund_orders,
                        "source": "demo",
                    }
                )
    return tuple(rows)


def _build_activity_metric_seeds() -> tuple[dict[str, object], ...]:
    """根据根目录测试活动文档构造可复核的模拟经营数据。"""
    rows: list[dict[str, object]] = []
    # 《春季上新活动复盘》明确记载活动 GMV 86,400 元、支付订单 312 单。
    spring_content = (
        (date(2026, 3, 8), 1000, 30, "10000.00", 0),
        (date(2026, 3, 9), 1100, 32, "8500.00", 0),
        (date(2026, 3, 10), 1150, 33, "9000.00", 1),
        (date(2026, 3, 11), 1200, 34, "9300.00", 0),
        (date(2026, 3, 12), 1180, 32, "8800.00", 1),
        (date(2026, 3, 13), 1080, 30, "8000.00", 0),
        (date(2026, 3, 14), 950, 29, "7700.00", 0),
    )
    spring_search = (
        (date(2026, 3, 8), 1800, 13, "3500.00", 0),
        (date(2026, 3, 9), 2100, 13, "3800.00", 0),
        (date(2026, 3, 10), 2500, 14, "4000.00", 1),
        (date(2026, 3, 11), 2700, 14, "4200.00", 0),
        (date(2026, 3, 12), 2600, 14, "4100.00", 0),
        (date(2026, 3, 13), 2200, 12, "3200.00", 0),
        (date(2026, 3, 14), 1900, 12, "2300.00", 0),
    )
    for channel, records in (
        ("内容种草（春季上新）", spring_content),
        ("搜索广告（春季上新）", spring_search),
    ):
        for metric_date, visitors, orders, gmv, refunds in records:
            rows.append(
                {
                    "metric_date": metric_date,
                    "channel": channel,
                    "product_id": "demo-product-001",
                    "visitor_count": visitors,
                    "paid_order_count": orders,
                    "paid_gmv": Decimal(gmv),
                    "refund_order_count": refunds,
                    "source": "demo",
                }
            )

    # 《七夕礼赠新品活动方案》包含礼盒与组合款，活动期数据体现组合款的较高客单价。
    for index in range(13):
        metric_date = date(2026, 8, 10) + timedelta(days=index)
        gift_box_orders = 8 + index % 4
        combo_orders = 5 + index % 3
        rows.extend(
            (
                {
                    "metric_date": metric_date,
                    "channel": "短视频内容（七夕）",
                    "product_id": "demo-product-004",
                    "visitor_count": 190 + index * 9,
                    "paid_order_count": gift_box_orders,
                    "paid_gmv": Decimal(gift_box_orders) * Decimal("239.00"),
                    "refund_order_count": 1 if index in (5, 11) else 0,
                    "source": "demo",
                },
                {
                    "metric_date": metric_date,
                    "channel": "直播间（七夕）",
                    "product_id": "demo-product-005",
                    "visitor_count": 150 + index * 8,
                    "paid_order_count": combo_orders,
                    # 组合款按活动“满 399 减 50”后的成交金额记录。
                    "paid_gmv": Decimal(combo_orders) * Decimal("349.00"),
                    "refund_order_count": 1 if index == 9 else 0,
                    "source": "demo",
                },
            )
        )

    # 《618 夏日焕新活动规则》仅给出节奏和优惠规则，以下数值为便于联调构造的模拟数据，
    # 不应被解释为文档中记载的真实业绩。正式期相对预热与返场呈现更高的流量和成交。
    for index in range(20):
        metric_date = date(2026, 6, 1) + timedelta(days=index)
        if index < 5:
            content_visitors, content_orders = 760 + index * 45, 24 + index
            search_visitors, search_orders = 1120 + index * 55, 22 + index
        elif index < 18:
            content_visitors, content_orders = 1480 + (index % 4) * 70, 66 + (index % 5) * 3
            search_visitors, search_orders = 1860 + (index % 3) * 90, 48 + (index % 4) * 2
        else:
            content_visitors, content_orders = 980 + (index - 18) * 40, 38 + index - 18
            search_visitors, search_orders = 1320 + (index - 18) * 60, 30 + index - 18
        rows.extend(
            (
                {
                    "metric_date": metric_date,
                    "channel": "内容种草（618）",
                    "product_id": "demo-product-001",
                    "visitor_count": content_visitors,
                    "paid_order_count": content_orders,
                    # 主推双肩包以活动优惠后的模拟支付金额入账。
                    "paid_gmv": Decimal(content_orders) * Decimal("279.00"),
                    "refund_order_count": 1 if index in (8, 16) else 0,
                    "source": "demo",
                },
                {
                    "metric_date": metric_date,
                    "channel": "搜索广告（618）",
                    "product_id": "demo-product-002",
                    "visitor_count": search_visitors,
                    "paid_order_count": search_orders,
                    "paid_gmv": Decimal(search_orders) * Decimal("139.00"),
                    "refund_order_count": 1 if index in (11, 19) else 0,
                    "source": "demo",
                },
            )
        )
    return tuple(rows)


ACTIVITY_METRIC_SEEDS = _build_activity_metric_seeds()
YEARLY_BASE_METRIC_SEEDS = _build_yearly_base_metric_seeds()
DEMO_METRIC_SEEDS = YEARLY_BASE_METRIC_SEEDS + ACTIVITY_METRIC_SEEDS


async def seed_demo_data() -> None:
    """在空业务库中初始化全年模拟数据；重复执行时不覆盖已有业务数据。"""
    await create_tables()
    async with SessionLocal() as session:
        # 指标知识独立于每日数据初始化；旧数据库升级后也能补齐 RAG 检索所需记录。
        await seed_metric_definitions(session)
        existing_metric = await session.scalar(
            select(DailyMetric.id).where(DailyMetric.source == "demo").limit(1)
        )
        existing_product_ids = set(
            await session.scalars(select(Product.id).where(Product.source == "demo"))
        )
        for product in DEMO_PRODUCTS:
            # 即使有人只清理了指标表，也不会重复写入已经存在的模拟商品。
            if product["id"] not in existing_product_ids:
                session.add(Product(**product, source="demo"))

        # DailyMetric 仅保存外键 ID，未声明 ORM 关系；先刷新商品以保证数据库外键成立。
        await session.flush()

        if existing_metric is None:
            session.add_all(DailyMetric(**metric) for metric in DEMO_METRIC_SEEDS)
        await session.commit()
        if existing_metric is None:
            print(f"已初始化 {len(DEMO_PRODUCTS)} 个模拟商品和 {len(DEMO_METRIC_SEEDS)} 条全年指标数据。")


async def reset_demo_business_data() -> None:
    """永久清空商品与日指标，再写入完整的 2026 年模拟业务数据。"""
    await create_tables()
    async with SessionLocal() as session:
        await session.execute(delete(DailyMetric))
        await session.execute(delete(Product))
        await session.flush()
        session.add_all(Product(**product, source="demo") for product in DEMO_PRODUCTS)
        await session.flush()
        session.add_all(DailyMetric(**metric) for metric in DEMO_METRIC_SEEDS)
        # 指标定义属于 RAG 知识，保留既有内容并补齐缺失的内置定义。
        await seed_metric_definitions(session)
        await session.commit()
        print(
            f"已清空业务商品与指标，并写入 {len(DEMO_PRODUCTS)} 个商品、"
            f"{len(DEMO_METRIC_SEEDS)} 条 2026 年模拟指标数据。"
        )


def _parse_arguments() -> argparse.Namespace:
    """区分容器启动时的安全增量初始化与人工确认后的业务数据重置。"""
    parser = argparse.ArgumentParser(description="管理本地模拟业务数据")
    parser.add_argument(
        "--reset-business-data",
        action="store_true",
        help="永久删除 products 和 daily_metrics 后重建全年模拟数据",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    if arguments.reset_business_data:
        asyncio.run(reset_demo_business_data())
    else:
        asyncio.run(seed_demo_data())
