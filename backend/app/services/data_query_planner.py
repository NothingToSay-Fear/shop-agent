"""数据查询子 Agent 使用的确定性规划规则与计划数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


_PRODUCT_MARKERS = ("商品", "单品", "SKU", "SPU", "货品")
_REFUND_MARKERS = ("退款", "退货", "售后")
_CAUSAL_MARKERS = ("为什么", "为何", "原因", "归因", "下滑原因", "下降原因")


@dataclass(frozen=True)
class DataQueryPlan:
    """受控数据库子 Agent 的计划输出，不包含可执行 SQL。"""

    metric_codes: tuple[str, ...] = ()
    periods: tuple[tuple[str, str, str], ...] = ()
    dimensions: tuple[str, ...] = ()
    unsupported_requirements: tuple[str, ...] = ()
    allow_metric_drilldown: bool = False

    @property
    def capability_notes(self) -> tuple[str, ...]:
        if not self.unsupported_requirements:
            return ()
        return (
            "当前受控经营数据仅提供 daily_metrics 的按天聚合指标，"
            + "；".join(self.unsupported_requirements)
            + "。这不是要求用户手工补数，而是当前尚未接入对应明细数据源。",
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "metric_codes": list(self.metric_codes),
            "periods": [
                {"label": label, "start_date": start_date, "end_date": end_date}
                for label, start_date, end_date in self.periods
            ],
            "dimensions": list(self.dimensions),
            "unsupported_requirements": list(self.unsupported_requirements),
            "allow_metric_drilldown": self.allow_metric_drilldown,
        }


def build_data_query_plan(
    constraints: Mapping[str, object],
    question: str = "",
    driver_graph: Mapping[str, tuple[str, ...]] | None = None,
) -> DataQueryPlan:
    """从已确认任务状态生成完整指标清单和数据能力边界。"""
    persisted = constraints.get("data_query_plan")
    if isinstance(persisted, dict):
        metric_codes = _string_tuple(persisted.get("metric_codes"))
        periods = _period_tuple(persisted.get("periods"))
        dimensions = _string_tuple(persisted.get("dimensions"))
        unsupported = _string_tuple(persisted.get("unsupported_requirements"))
        allow_metric_drilldown = bool(persisted.get("allow_metric_drilldown", False))
        if metric_codes or periods or dimensions or unsupported:
            return DataQueryPlan(
                metric_codes,
                periods,
                dimensions,
                unsupported,
                allow_metric_drilldown,
            )

    metric_codes = _string_tuple(constraints.get("metrics"))
    periods = _period_tuple(constraints.get("periods"))
    lowered = question.lower()
    allow_metric_drilldown = _allows_metric_drilldown(constraints, lowered)
    if allow_metric_drilldown:
        metric_codes = _expand_causal_driver_metrics(metric_codes, driver_graph or {})
    wants_product_refunds = any(marker.lower() in lowered for marker in _PRODUCT_MARKERS) and any(
        marker in lowered for marker in _REFUND_MARKERS
    )
    if wants_product_refunds:
        return DataQueryPlan(
            metric_codes=metric_codes,
            periods=periods,
            dimensions=("product",),
            unsupported_requirements=(
                "无法按商品/SKU 排序高退款商品，也无法关联订单退款原因或活动规则",
            ),
            allow_metric_drilldown=allow_metric_drilldown,
        )
    return DataQueryPlan(
        metric_codes=metric_codes,
        periods=periods,
        allow_metric_drilldown=allow_metric_drilldown,
    )


def _allows_metric_drilldown(constraints: Mapping[str, object], lowered_question: str) -> bool:
    execution_intent = constraints.get("execution_intent")
    if isinstance(execution_intent, dict):
        return execution_intent.get("allow_metric_drilldown") is True
    return constraints.get("analysis_goal") == "经营归因" or any(
        marker in lowered_question for marker in _CAUSAL_MARKERS
    )


def _expand_causal_driver_metrics(
    metric_codes: tuple[str, ...], driver_graph: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """按数据库图谱补齐直接驱动指标，不在代码中固化业务指标关系。"""
    driver_codes = [
        driver_code
        for metric_code in metric_codes
        for driver_code in driver_graph.get(metric_code, ())
    ]
    return tuple(dict.fromkeys((*metric_codes, *driver_codes)))


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _period_tuple(value: object) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(value, list):
        return ()
    periods: list[tuple[str, str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        start_date = item.get("start_date")
        end_date = item.get("end_date")
        if all(isinstance(part, str) and part for part in (label, start_date, end_date)):
            periods.append((label, start_date, end_date))
    return tuple(periods)
