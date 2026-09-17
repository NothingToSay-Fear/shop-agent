"""主 Agent 的受限委派规划：决定是否调用复盘子 Agent。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from app.models import ConversationTask
from app.services.conversation_tasks import TASK_HYBRID_ANALYSIS, TASK_METRIC_COMPARISON, TASK_REVIEW


@dataclass(frozen=True)
class DelegationPlan:
    kind: str
    route_mode: str
    use_review_agent: bool
    batches: tuple[tuple[str, ...], ...]
    actions: tuple["PlannedAction", ...] = ()
    budget: Mapping[str, int] = field(
        default_factory=lambda: {"max_actions": 8, "max_replans": 2, "max_tool_calls": 5}
    )

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
    """主 Agent 的受控 Planner；只产出可验证 Action，不执行工具或编写结论。"""

    def plan(self, task: ConversationTask) -> DelegationPlan:
        """从任务快照的多维意图生成 Action，而非由单一 task_type 互斥分流。"""
        constraints = getattr(task, "effective_constraints", {}) or {}
        intent = constraints.get("execution_intent") if isinstance(constraints, dict) else None
        if not isinstance(intent, dict):
            intent = _legacy_execution_intent(task.task_type)
        sources = tuple(
            source
            for source in intent.get("evidence_sources", [])
            if source in {"metrics", "knowledge"}
        )
        operation = str(intent.get("operation") or "lookup")
        use_review_agent = operation in {"review", "causal_analysis"}
        kind = {
            "comparison": "comparison",
            "causal_analysis": "analysis",
            "review": "review",
        }.get(operation, "lookup")

        actions: list[PlannedAction] = [_confirm_action()]
        evidence_keys: list[str] = []
        if "metrics" in sources:
            actions.append(_metric_action())
            evidence_keys.append("query_metrics")
        if "knowledge" in sources:
            actions.append(_knowledge_action())
            evidence_keys.append("query_knowledge")
        actions.append(_evaluation_action(tuple(evidence_keys)))
        if use_review_agent:
            actions.append(
                PlannedAction(
                    key="review_evidence",
                    title="基于已验证证据完成复盘",
                    action_type="review",
                    depends_on=("evaluate_evidence",),
                    expected_output="区分事实、资料依据、待验证假设和建议",
                )
            )
            actions.append(_synthesis_action(("review_evidence",)))
        else:
            actions.append(_synthesis_action(("evaluate_evidence",)))

        route_mode = "hybrid" if len(sources) == 2 else ("knowledge" if sources == ("knowledge",) else "metrics")
        first_batch = tuple(
            name for source, name in (("metrics", "data_query"), ("knowledge", "knowledge_retrieval")) if source in sources
        )
        batches = (first_batch, ("evidence_evaluation",))
        if use_review_agent:
            batches += (("review",),)
        return DelegationPlan(kind, route_mode, use_review_agent, batches, tuple(actions))

    def replan(
        self,
        plan: DelegationPlan,
        *,
        next_metric_codes: tuple[str, ...] = (),
        reason: str,
    ) -> tuple["PlannedAction", ...]:
        """只接受数据子 Agent 提出的已登记指标，生成追加调查动作。

        主 Agent 不接受自由文本工具调用；后续动作仍受 Action Schema、预算和能力目录限制。
        """
        if not next_metric_codes or plan.budget.get("max_replans", 0) <= 0:
            return ()
        return (
            PlannedAction(
                key="followup_metrics",
                title="补充验证驱动指标",
                action_type="query_metrics",
                tool_name="query_metric_rag",
                depends_on=("evaluate_evidence",),
                action_input={"metric_codes": list(next_metric_codes), "reason": reason},
                expected_output="获得下一层已登记驱动指标的同口径数据",
                capability_requirement={"source": "daily_metrics", "granularity": "day"},
            ),
            PlannedAction(
                key="followup_evaluation",
                title="评估补充证据",
                action_type="evaluate_evidence",
                depends_on=("followup_metrics",),
                expected_output="判断是否足以形成受控结论",
            ),
        )

    def _plan_metrics(self, kind: str, task: ConversationTask) -> DelegationPlan:
        actions = (
            _confirm_action(),
            _metric_action(),
            _evaluation_action(("query_metrics",)),
            _synthesis_action(("evaluate_evidence",)),
        )
        return DelegationPlan(kind, "metrics", False, (("data_query",),), actions)

    def _plan_analysis(self, kind: str, task: ConversationTask, *, include_review: bool) -> DelegationPlan:
        actions: list[PlannedAction] = [_confirm_action(), _metric_action(), _knowledge_action()]
        actions.append(_evaluation_action(("query_metrics", "query_knowledge")))
        if include_review:
            actions.append(
                PlannedAction(
                    key="review_evidence",
                    title="基于已验证证据完成复盘",
                    action_type="review",
                    depends_on=("evaluate_evidence",),
                    expected_output="区分事实、资料依据、待验证假设和建议",
                )
            )
            actions.append(_synthesis_action(("review_evidence",)))
        else:
            actions.append(_synthesis_action(("evaluate_evidence",)))
        return DelegationPlan(
            kind,
            "hybrid",
            include_review,
            (("data_query", "knowledge_retrieval"), ("evidence_evaluation",), ("review",)),
            tuple(actions),
        )


@dataclass(frozen=True)
class PlannedAction:
    """主 Agent 唯一允许交给执行器的结构化动作。"""

    key: str
    title: str
    action_type: str
    tool_name: str | None = None
    depends_on: tuple[str, ...] = ()
    action_input: Mapping[str, object] = field(default_factory=dict)
    expected_output: str | None = None
    capability_requirement: Mapping[str, object] = field(default_factory=dict)
    condition: Mapping[str, object] = field(default_factory=dict)


def _legacy_execution_intent(task_type: str) -> dict[str, object]:
    """兼容尚未写入 execution_intent 的历史任务与单元测试构造对象。"""
    if task_type == TASK_REVIEW:
        return {"operation": "review", "evidence_sources": ["metrics", "knowledge"]}
    if task_type == TASK_HYBRID_ANALYSIS:
        return {"operation": "causal_analysis", "evidence_sources": ["metrics", "knowledge"]}
    if task_type == TASK_METRIC_COMPARISON:
        return {"operation": "comparison", "evidence_sources": ["metrics"]}
    return {"operation": "lookup", "evidence_sources": ["metrics"]}


def _confirm_action() -> PlannedAction:
    return PlannedAction(
        key="confirm_constraints",
        title="确认任务约束与可用能力",
        action_type="confirm_constraints",
        expected_output="确认时间、指标、目标和能力边界",
    )


def _metric_action() -> PlannedAction:
    return PlannedAction(
        key="query_metrics",
        title="查询经营指标",
        action_type="query_metrics",
        tool_name="query_metric_rag",
        depends_on=("confirm_constraints",),
        action_input={"use_data_query_agent": True},
        expected_output="获得指定范围内的受控经营指标",
        capability_requirement={"source": "daily_metrics", "granularity": "day"},
    )


def _knowledge_action() -> PlannedAction:
    return PlannedAction(
        key="query_knowledge",
        title="检索活动与运营资料",
        action_type="query_knowledge",
        tool_name="query_knowledge_rag",
        depends_on=("confirm_constraints",),
        expected_output="获得可引用的活动规则或运营资料",
        capability_requirement={"source": "knowledge_base"},
    )


def _evaluation_action(depends_on: tuple[str, ...]) -> PlannedAction:
    return PlannedAction(
        key="evaluate_evidence",
        title="评估证据充分性并决定是否下钻",
        action_type="evaluate_evidence",
        depends_on=depends_on,
        expected_output="判断证据是否足够，或提出受控的下一步数据动作",
    )


def _synthesis_action(depends_on: tuple[str, ...]) -> PlannedAction:
    return PlannedAction(
        key="synthesize",
        title="形成受控结论",
        action_type="synthesize",
        depends_on=depends_on,
        expected_output="只基于已完成动作中的证据生成回答",
    )
