"""将路由结果转为可验证的受控工具执行计划。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.agent.tools.registry import ToolName, get_tool_specification
from app.services.intent_router import RetrievalRoute

if TYPE_CHECKING:
    from app.agent.tools.tracker import ToolCallAudit


ROUTE_TOOL_NAMES: dict[str, tuple[ToolName, ...]] = {
    "metrics": ("query_metric_rag",),
    "knowledge": ("query_knowledge_rag",),
    "web": ("search_web",),
    "hybrid": ("query_metric_rag", "query_knowledge_rag"),
    "web_hybrid": ("query_metric_rag", "query_knowledge_rag", "search_web"),
}
FINAL_TOOL_STATUSES = frozenset({"success", "empty", "skipped", "failed"})


@dataclass(frozen=True)
class ExecutionPlan:
    """某次路由必须完成的工具清单；工具失败不阻断其他工具执行。"""

    route_mode: str
    required_tools: tuple[ToolName, ...]

    @property
    def summary(self) -> str:
        """供提示词和日志使用的无用户原文计划摘要。"""
        return "、".join(self.required_tools) if self.required_tools else "无需检索工具"


@dataclass(frozen=True)
class PlanValidation:
    """执行计划完整性与成功工具引用完整性的规则校验结果。"""

    missing_tools: tuple[ToolName, ...]
    invalid_reference_tools: tuple[ToolName, ...]
    exceeded_call_limit_tools: tuple[ToolName, ...]

    @property
    def passed(self) -> bool:
        return not (
            self.missing_tools or self.invalid_reference_tools or self.exceeded_call_limit_tools
        )

    @property
    def summary(self) -> str:
        """返回稳定错误码组成的摘要，避免写入工具原始输出。"""
        issues: list[str] = []
        if self.missing_tools:
            issues.append("missing_required_tool:" + ",".join(self.missing_tools))
        if self.invalid_reference_tools:
            issues.append("invalid_tool_references:" + ",".join(self.invalid_reference_tools))
        if self.exceeded_call_limit_tools:
            issues.append("tool_call_limit_exceeded:" + ",".join(self.exceeded_call_limit_tools))
        return "；".join(issues) if issues else "passed"


def build_execution_plan(route: RetrievalRoute) -> ExecutionPlan:
    """将每种固定路由映射为代码级必调工具，禁止由模型自行删减。"""
    plan = build_execution_plan_for_mode(route.mode)
    if plan is None:
        raise ValueError(f"不支持的执行计划路由：{route.mode}")
    return plan


def build_execution_plan_for_mode(route_mode: str | None) -> ExecutionPlan | None:
    """为审计展示等只有已持久化路由名称的场景恢复执行计划。"""
    required_tools = ROUTE_TOOL_NAMES.get(route_mode or "")
    if required_tools is None:
        return None
    return ExecutionPlan(route_mode=route_mode or "", required_tools=required_tools)


def validate_execution_plan(
    plan: ExecutionPlan, tool_calls: list[ToolCallAudit]
) -> PlanValidation:
    """确认必调工具均有终态记录，成功结果也必须带符合类型的引用 ID。"""
    missing_tools: list[ToolName] = []
    invalid_reference_tools: list[ToolName] = []
    exceeded_call_limit_tools: list[ToolName] = []

    for tool_name in plan.required_tools:
        calls = [call for call in tool_calls if call.tool_name == tool_name]
        if not any(call.status in FINAL_TOOL_STATUSES for call in calls):
            missing_tools.append(tool_name)
            continue

        specification = get_tool_specification(tool_name)
        real_calls = [call for call in calls if call.error_code != "tool_call_limit_reached"]
        if len(real_calls) > specification.max_calls_per_run:
            exceeded_call_limit_tools.append(tool_name)

        for call in calls:
            if call.status != "success":
                continue
            if not call.reference_ids or not all(
                reference_id.startswith(specification.reference_prefixes)
                for reference_id in call.reference_ids
            ):
                invalid_reference_tools.append(tool_name)
                break

    return PlanValidation(
        missing_tools=tuple(missing_tools),
        invalid_reference_tools=tuple(invalid_reference_tools),
        exceeded_call_limit_tools=tuple(exceeded_call_limit_tools),
    )
