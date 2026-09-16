"""受控数据库子 Agent：规划并执行单轮经营数据查询。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from langchain_core.tools import BaseTool

from app.services.data_query_planner import DataQueryPlan, build_data_query_plan

if TYPE_CHECKING:
    from app.services.conversation_context import ConversationContextSnapshot


DATA_QUERY_AGENT_NAME = "data_query_agent"


@dataclass
class DataQueryAgent:
    """只拥有数据查询计划与只读指标工具，不拥有任意 SQL 或写入能力。

    主工作流将指标工具委派给该 Agent。它先基于任务有效约束规划，
    再以结构化指标和 ISO 时间范围调用工具；没有结构化计划时不替代
    原有指标 RAG 的自由问答降级路径。
    """

    constraints: Mapping[str, object]
    question: str
    _plan: DataQueryPlan | None = field(default=None, init=False, repr=False)

    def plan(self) -> DataQueryPlan:
        """生成可审计、无 SQL 的数据查询计划。"""
        if self._plan is None:
            self._plan = build_data_query_plan(self.constraints, self.question)
        return self._plan

    async def execute(
        self,
        metric_tool: BaseTool,
        retrieval_question: str,
        conversation_context: ConversationContextSnapshot,
    ) -> DataQueryPlan:
        """执行唯一获授权的只读指标工具，并返回实际使用的计划。"""
        plan = self.plan()
        payload: dict[str, object] = {
            "question": retrieval_question,
            "start_date": conversation_context.start_date,
            "end_date": conversation_context.end_date,
        }
        if plan.metric_codes:
            payload["metric_codes"] = list(plan.metric_codes)
        if plan.periods:
            payload["query_periods"] = [
                {"label": label, "start_date": start_date, "end_date": end_date}
                for label, start_date, end_date in plan.periods
            ]
        if plan.capability_notes:
            payload["capability_notes"] = list(plan.capability_notes)
        await metric_tool.ainvoke(payload)
        return plan
