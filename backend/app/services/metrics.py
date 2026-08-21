"""模拟经营指标的查询与文本摘要。"""

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DailyMetric


@dataclass(frozen=True)
class MetricPeriod:
    """一个统计周期内聚合后的核心经营指标。"""

    start_date: date
    end_date: date
    gmv: float
    paid_orders: int
    visitors: int
    refund_orders: int

    @property
    def conversion_rate(self) -> float:
        """支付订单数除以访客数，访客为零时返回零。"""
        return self.paid_orders / self.visitors if self.visitors else 0


@dataclass(frozen=True)
class MetricsOverview:
    """当前周期与上一周期的对比结果，可同时供 API 与 Agent 使用。"""

    current: MetricPeriod
    previous: MetricPeriod

    def as_dict(self) -> dict[str, object]:
        """将日期与计算属性转换为适合 JSON 响应的结构。"""
        return {
            "current": {**asdict(self.current), "conversion_rate": self.current.conversion_rate},
            "previous": {**asdict(self.previous), "conversion_rate": self.previous.conversion_rate},
            "gmv_change_rate": _change_rate(self.current.gmv, self.previous.gmv),
            "visitor_change_rate": _change_rate(self.current.visitors, self.previous.visitors),
        }

    def to_agent_context(self) -> str:
        """将可追溯的模拟数据压缩为 Agent 可直接引用的上下文。"""
        gmv_change = _change_rate(self.current.gmv, self.previous.gmv)
        visitor_change = _change_rate(self.current.visitors, self.previous.visitors)
        return (
            f"数据来源：内置模拟经营数据。当前周期：{self.current.start_date} 至 {self.current.end_date}；"
            f"GMV {self.current.gmv:,.2f} 元，支付订单 {self.current.paid_orders}，"
            f"访客 {self.current.visitors}，支付转化率 {self.current.conversion_rate:.2%}，"
            f"退款订单 {self.current.refund_orders}。\n"
            f"对比周期：{self.previous.start_date} 至 {self.previous.end_date}；"
            f"GMV {self.previous.gmv:,.2f} 元，支付订单 {self.previous.paid_orders}，"
            f"访客 {self.previous.visitors}，支付转化率 {self.previous.conversion_rate:.2%}。\n"
            f"GMV 环比 {gmv_change:+.2%}，访客环比 {visitor_change:+.2%}。"
        )


async def get_metrics_overview(session: AsyncSession) -> MetricsOverview | None:
    """根据数据库中最新的 14 天模拟数据计算最近两周的环比摘要。"""
    latest_date = await session.scalar(
        select(func.max(DailyMetric.metric_date)).where(DailyMetric.source == "demo")
    )
    if latest_date is None:
        return None

    current_start = latest_date - timedelta(days=6)
    previous_end = current_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=6)
    current = await _aggregate_period(session, current_start, latest_date)
    previous = await _aggregate_period(session, previous_start, previous_end)
    return MetricsOverview(current=current, previous=previous)


async def _aggregate_period(
    session: AsyncSession, start_date: date, end_date: date
) -> MetricPeriod:
    """聚合一个日期区间的模拟经营数据。"""
    result = await session.execute(
        select(
            func.coalesce(func.sum(DailyMetric.paid_gmv), 0),
            func.coalesce(func.sum(DailyMetric.paid_order_count), 0),
            func.coalesce(func.sum(DailyMetric.visitor_count), 0),
            func.coalesce(func.sum(DailyMetric.refund_order_count), 0),
        ).where(
            DailyMetric.source == "demo",
            DailyMetric.metric_date.between(start_date, end_date),
        )
    )
    gmv, paid_orders, visitors, refund_orders = result.one()
    return MetricPeriod(
        start_date=start_date,
        end_date=end_date,
        gmv=float(gmv or Decimal("0")),
        paid_orders=int(paid_orders or 0),
        visitors=int(visitors or 0),
        refund_orders=int(refund_orders or 0),
    )


def _change_rate(current: float | int, previous: float | int) -> float:
    """计算环比变化率；上一周期为零时返回零以避免误导性无穷值。"""
    return (current - previous) / previous if previous else 0
