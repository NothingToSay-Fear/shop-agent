"""从数据库读取和初始化可审计的指标归因图谱。"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import MetricAnalysisDriver


# 仅用于首次部署/演示数据初始化；运行期的查询计划只读取数据库中的边。
METRIC_ANALYSIS_DRIVER_SEEDS: tuple[dict[str, object], ...] = (
    {"metric_code": "paid_gmv", "driver_metric_code": "paid_order_count", "display_order": 10},
    {"metric_code": "paid_gmv", "driver_metric_code": "visitor_count", "display_order": 20},
    {"metric_code": "paid_gmv", "driver_metric_code": "conversion_rate", "display_order": 30},
    {"metric_code": "paid_gmv", "driver_metric_code": "average_order_value", "display_order": 40},
    {"metric_code": "paid_order_count", "driver_metric_code": "visitor_count", "display_order": 10},
    {"metric_code": "paid_order_count", "driver_metric_code": "conversion_rate", "display_order": 20},
    {"metric_code": "refund_rate", "driver_metric_code": "refund_order_count", "display_order": 10},
    {"metric_code": "refund_rate", "driver_metric_code": "paid_order_count", "display_order": 20},
)


async def load_metric_analysis_driver_graph(
    session: AsyncSession,
) -> dict[str, tuple[str, ...]]:
    """读取当前启用的直接驱动边，保持运营配置的展示顺序。"""
    rows = list(
        await session.scalars(
            select(MetricAnalysisDriver)
            .where(
                MetricAnalysisDriver.enabled.is_(True),
                MetricAnalysisDriver.relationship_type == "driver",
            )
            .order_by(MetricAnalysisDriver.metric_code, MetricAnalysisDriver.display_order)
        )
    )
    graph: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        graph[row.metric_code].append(row.driver_metric_code)
    return {metric_code: tuple(driver_codes) for metric_code, driver_codes in graph.items()}


async def seed_metric_analysis_drivers(session: AsyncSession) -> None:
    """幂等补齐内置图谱边，不覆盖运营人员已维护的边。"""
    existing_edges = set(
        await session.execute(
            select(MetricAnalysisDriver.metric_code, MetricAnalysisDriver.driver_metric_code)
        )
    )
    for edge in METRIC_ANALYSIS_DRIVER_SEEDS:
        key = (str(edge["metric_code"]), str(edge["driver_metric_code"]))
        if key not in existing_edges:
            session.add(MetricAnalysisDriver(**edge))
