"""主 Agent 的受限委派规划：决定是否调用复盘子 Agent。"""

from __future__ import annotations

from dataclasses import dataclass

from app.models import ConversationTask
from app.services.conversation_tasks import TASK_HYBRID_ANALYSIS, TASK_METRIC_COMPARISON, TASK_REVIEW


@dataclass(frozen=True)
class DelegationPlan:
    kind: str
    route_mode: str
    use_review_agent: bool
    batches: tuple[tuple[str, ...], ...]

    @property
    def output_contract(self) -> str:
        if self.kind == "lookup":
            return "这是纯指标查询：只输出指标、时间范围、数值、口径或数据缺口；不要给复盘、归因、建议或下一步。"
        if self.kind == "comparison":
            return "这是指标对比：只输出对比表、可比性提示和受控差异；不要给归因或优化建议。"
        return (
            "这是指标归因/复盘任务：先基于已查询的指标说明变化由流量、转化、订单或客单价中的"
            "哪些因素驱动；再区分数据事实、资料依据、待验证假设和建议动作。不得把未接入的"
            "渠道、商品、活动规则等明细推测为根因；不得要求用户补充本轮数据子 Agent 已查询的指标。"
        )


class MainAgentOrchestrator:
    """主 Agent 的任务拆解器；只输出有限委派计划，不执行数据库或复盘。"""

    def plan(self, task: ConversationTask) -> DelegationPlan:
        if task.task_type == TASK_REVIEW:
            return DelegationPlan("review", "hybrid", True, (("data_query", "knowledge_retrieval"), ("review",)))
        if task.task_type == TASK_HYBRID_ANALYSIS:
            return DelegationPlan("analysis", "hybrid", True, (("data_query", "knowledge_retrieval"), ("review",)))
        if task.task_type == TASK_METRIC_COMPARISON:
            return DelegationPlan("comparison", "metrics", False, (("data_query",),))
        return DelegationPlan("lookup", "metrics", False, (("data_query",),))
