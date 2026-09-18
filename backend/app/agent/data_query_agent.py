"""受控数据库子 Agent：规划并执行单轮经营数据查询。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from langchain_core.tools import BaseTool

from app.services.analytics.data_query_planner import DataQueryPlan, build_data_query_plan

if TYPE_CHECKING:
    from app.agent.tools.tracker import AgentToolTracker
    from app.services.conversations.context import ConversationContextSnapshot


DATA_QUERY_AGENT_NAME = "data_query_agent"


@dataclass(frozen=True)
class DataQueryObservation:
    """数据库子 Agent 对已验证查询结果的受控观察。"""

    sufficient: bool
    reason: str
    next_metric_codes: tuple[str, ...] = ()
    blocked_by_capability: tuple[str, ...] = ()


@dataclass
class DataQueryAgent:
    """只拥有数据查询计划与只读指标工具，不拥有任意 SQL 或写入能力。

    主工作流将指标工具委派给该 Agent。它先基于任务有效约束规划，
    再以结构化指标和 ISO 时间范围调用工具；没有结构化计划时不替代
    原有指标 RAG 的自由问答降级路径。
    """

    constraints: Mapping[str, object]
    question: str
    driver_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    _plan: DataQueryPlan | None = field(default=None, init=False, repr=False)

    def plan(self) -> DataQueryPlan:
        """生成可审计、无 SQL 的数据查询计划。"""
        if self._plan is None:
            self._plan = build_data_query_plan(self.constraints, self.question, self.driver_graph)
        return self._plan

    async def execute(
        self,
        metric_tool: BaseTool,
        retrieval_question: str,
        conversation_context: ConversationContextSnapshot,
        metric_codes: tuple[str, ...] | None = None,
    ) -> DataQueryPlan:
        """执行唯一获授权的只读指标工具，并返回实际使用的计划。"""
        plan = self.plan()
        payload: dict[str, object] = {
            "question": retrieval_question,
            "start_date": conversation_context.start_date,
            "end_date": conversation_context.end_date,
        }
        selected_codes = metric_codes or plan.metric_codes
        if selected_codes:
            payload["metric_codes"] = list(selected_codes)
        if plan.periods:
            payload["query_periods"] = [
                {"label": label, "start_date": start_date, "end_date": end_date}
                for label, start_date, end_date in plan.periods
            ]
        if plan.dimensions:
            payload["dimensions"] = list(plan.dimensions)
        if plan.capability_notes:
            payload["capability_notes"] = list(plan.capability_notes)
        await metric_tool.ainvoke(payload)
        return plan

    def observe(self, tracker: "AgentToolTracker") -> DataQueryObservation:
        """根据已查询指标和能力目录判断是否值得继续调查。

        它只提出图谱中尚未查询的下一层指标；是否追加动作由主 Agent 决定。
        """
        plan = self.plan()
        if tracker.metric_context is None:
            return DataQueryObservation(False, "未获得可用经营指标，不能继续做数据归因")
        if not plan.allow_metric_drilldown:
            return DataQueryObservation(
                True,
                "当前计划仅允许按用户指定口径查询，不授权继续下钻驱动指标",
            )
        queried = set(tracker.metric_context.metric_codes)
        candidates = tuple(
            dict.fromkeys(
                driver
                for metric in queried
                for driver in self.driver_graph.get(metric, ())
                if driver not in queried
            )
        )
        if candidates:
            return DataQueryObservation(False, "一级证据已获取，可继续验证下一层驱动", candidates)
        return DataQueryObservation(True, "当前可用指标及其已登记驱动已完成查询")
