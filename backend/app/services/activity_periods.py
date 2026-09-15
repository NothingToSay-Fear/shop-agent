"""集中维护预置测试活动名称与其可查询日期范围。"""

from __future__ import annotations

from datetime import date

from app.services.date_ranges import DEMO_DATA_YEAR

# 只有用户明确提到这些活动名称时，才可将其映射为受控查询日期。
ACTIVITY_PERIODS = {
    "618": (("618", "六一八"), date(DEMO_DATA_YEAR, 6, 1), date(DEMO_DATA_YEAR, 6, 20)),
    "七夕": (("七夕",), date(DEMO_DATA_YEAR, 8, 10), date(DEMO_DATA_YEAR, 8, 22)),
    "春季上新": (("春季上新",), date(DEMO_DATA_YEAR, 3, 8), date(DEMO_DATA_YEAR, 3, 14)),
}
