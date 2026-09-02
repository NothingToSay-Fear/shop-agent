"""统一解析可安全进入受控查询的日期范围。"""

from __future__ import annotations

import re
from datetime import date

# 演示经营数据当前覆盖 2026 年。无年份的日期表达按该业务数据年度解释，避免依赖部署机器日期。
DEMO_DATA_YEAR = 2026
_SEPARATOR = r"(?:至|到|~|-|–|—)"
_CHINESE_RANGE_PATTERN = re.compile(
    rf"(?P<year>20\d{{2}})\s*年\s*(?P<month>\d{{1,2}})\s*月\s*(?P<day>\d{{1,2}})\s*日?"
    rf"\s*{_SEPARATOR}\s*(?:(?P<end_year>20\d{{2}})\s*年\s*)?"
    r"(?P<end_month>\d{1,2})\s*月\s*(?P<end_day>\d{1,2})\s*日?"
)
_ISO_RANGE_PATTERN = re.compile(
    rf"(?P<year>20\d{{2}})\s*[-/.]\s*(?P<month>\d{{1,2}})\s*[-/.]\s*(?P<day>\d{{1,2}})"
    rf"\s*{_SEPARATOR}\s*(?:(?P<end_year>20\d{{2}})\s*[-/.]\s*)?"
    r"(?P<end_month>\d{1,2})\s*[-/.]\s*(?P<end_day>\d{1,2})"
)
_MONTH_DAY_RANGE_PATTERN = re.compile(
    rf"(?P<month>\d{{1,2}})\s*月\s*(?P<day>\d{{1,2}})\s*日?\s*{_SEPARATOR}\s*"
    r"(?P<end_month>\d{1,2})\s*月\s*(?P<end_day>\d{1,2})\s*日?"
)
_SLASH_RANGE_PATTERN = re.compile(
    rf"(?P<month>\d{{1,2}})\s*/\s*(?P<day>\d{{1,2}})\s*{_SEPARATOR}\s*"
    r"(?P<end_month>\d{1,2})\s*/\s*(?P<end_day>\d{1,2})"
)


def parse_explicit_date_range(
    text: str, default_year: int = DEMO_DATA_YEAR
) -> tuple[date, date] | None:
    """识别中文、ISO 和月/日简写区间，失败时不返回不可信日期。"""
    for pattern, use_default_year in (
        (_CHINESE_RANGE_PATTERN, False),
        (_ISO_RANGE_PATTERN, False),
        (_MONTH_DAY_RANGE_PATTERN, True),
        (_SLASH_RANGE_PATTERN, True),
    ):
        match = pattern.search(text)
        if match is None:
            continue
        period = _build_period(match.groupdict(), default_year, use_default_year)
        if period is not None:
            return period
    return None


def _build_period(
    values: dict[str, str | None], default_year: int, use_default_year: bool
) -> tuple[date, date] | None:
    try:
        start_year = default_year if use_default_year else int(values["year"] or default_year)
        end_year = int(values.get("end_year") or start_year)
        start_date = date(start_year, int(values["month"] or 0), int(values["day"] or 0))
        end_date = date(end_year, int(values["end_month"] or 0), int(values["end_day"] or 0))
    except ValueError:
        return None
    if use_default_year and end_date < start_date:
        try:
            end_date = date(start_year + 1, end_date.month, end_date.day)
        except ValueError:
            return None
    return (start_date, end_date) if start_date <= end_date else None
