"""初始化可重复执行的本地模拟经营数据。"""

import asyncio
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import select

from app.database import SessionLocal, create_tables
from app.models import DailyMetric, Product

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
)


async def seed_demo_data() -> None:
    """仅在没有模拟数据时写入商品和 14 天经营指标，确保重复执行安全。"""
    await create_tables()
    async with SessionLocal() as session:
        existing_metric = await session.scalar(
            select(DailyMetric.id).where(DailyMetric.source == "demo").limit(1)
        )
        if existing_metric is not None:
            print("模拟经营数据已存在，跳过初始化。")
            return

        existing_product_ids = set(
            await session.scalars(select(Product.id).where(Product.source == "demo"))
        )
        for product in DEMO_PRODUCTS:
            # 即使有人只清理了指标表，也不会重复写入已经存在的模拟商品。
            if product["id"] not in existing_product_ids:
                session.add(Product(**product, source="demo"))

        # DailyMetric 仅保存外键 ID，未声明 ORM 关系；先刷新商品以保证数据库外键成立。
        await session.flush()

        today = date.today()
        channels = (("内容种草", 1.0), ("搜索广告", 0.72))
        for days_ago in range(13, -1, -1):
            metric_date = today - timedelta(days=days_ago)
            # 最近一周访客下降，转化率略有提升，便于演示有依据的经营诊断。
            is_current_week = days_ago <= 6
            for product_index, product in enumerate(DEMO_PRODUCTS):
                for channel_index, (channel, channel_factor) in enumerate(channels):
                    base_visitors = 620 + product_index * 95 + (days_ago % 4) * 28
                    visitors = int(base_visitors * channel_factor * (0.88 if is_current_week else 1))
                    conversion = 0.035 + product_index * 0.003 + (0.003 if is_current_week else 0)
                    paid_orders = max(1, round(visitors * conversion))
                    refund_orders = 1 if (days_ago + product_index + channel_index) % 6 == 0 else 0
                    session.add(
                        DailyMetric(
                            metric_date=metric_date,
                            channel=channel,
                            product_id=product["id"],
                            visitor_count=visitors,
                            paid_order_count=paid_orders,
                            paid_gmv=Decimal(paid_orders) * product["price"],
                            refund_order_count=refund_orders,
                            source="demo",
                        )
                    )
        await session.commit()
        print("已初始化 3 个模拟商品和 14 天经营指标数据。")


if __name__ == "__main__":
    asyncio.run(seed_demo_data())
