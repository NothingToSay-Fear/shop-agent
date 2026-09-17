"""从数据库读取和初始化可审计的指标归因图谱。"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import MetricAnalysisDriver


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
