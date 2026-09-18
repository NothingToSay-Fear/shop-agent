"""集中维护预置测试活动名称与其可查询日期范围。"""

from __future__ import annotations

from datetime import date

from app.services.analytics.date_ranges import DEMO_DATA_YEAR

# 只有用户明确提到这些活动名称时，才可将其映射为受控查询日期。
ACTIVITY_PERIODS = {
    "618": (("618", "六一八"), date(DEMO_DATA_YEAR, 6, 1), date(DEMO_DATA_YEAR, 6, 20)),
    "七夕": (("七夕",), date(DEMO_DATA_YEAR, 8, 10), date(DEMO_DATA_YEAR, 8, 22)),
    "春季上新": (("春季上新",), date(DEMO_DATA_YEAR, 3, 8), date(DEMO_DATA_YEAR, 3, 14)),
}


def resolve_activity_periods(question: str) -> tuple[tuple[str, date, date], ...]:
    """按用户原文顺序解析已登记活动，供任务层和指标层共享。"""
    lowered = question.lower()
    matched: list[tuple[int, tuple[str, date, date]]] = []
    for activity, (aliases, start_date, end_date) in ACTIVITY_PERIODS.items():
        positions = [lowered.find(alias.lower()) for alias in aliases if lowered.find(alias.lower()) >= 0]
        if positions:
            matched.append((min(positions), (activity, start_date, end_date)))
    matched.sort(key=lambda item: item[0])

    seen: set[tuple[date, date]] = set()
    periods: list[tuple[str, date, date]] = []
    for activity, start_date, end_date in (item for _, item in matched):
        if (start_date, end_date) in seen:
            continue
        seen.add((start_date, end_date))
        periods.append((activity, start_date, end_date))
    return tuple(periods)
