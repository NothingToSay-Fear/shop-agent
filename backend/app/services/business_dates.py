"""统一处理经营数据查询使用的业务日期。"""

from datetime import date, datetime
from zoneinfo import ZoneInfo


BUSINESS_TIMEZONE = ZoneInfo("Asia/Shanghai")


def current_business_date() -> date:
    """返回上海时区的当前自然日，避免容器时区影响经营数据查询。"""
    return datetime.now(BUSINESS_TIMEZONE).date()
