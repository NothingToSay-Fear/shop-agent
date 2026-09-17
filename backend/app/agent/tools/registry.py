"""受控工具的运行时注册信息。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ToolName = Literal["query_metric_rag", "query_knowledge_rag", "search_web"]


@dataclass(frozen=True)
class ToolSpecification:
    """声明工具的稳定名称、调用上限和成功结果应具备的引用类型。"""

    name: ToolName
    max_calls_per_run: int
    reference_prefixes: tuple[str, ...]


TOOL_REGISTRY: dict[ToolName, ToolSpecification] = {
    "query_metric_rag": ToolSpecification(
        name="query_metric_rag", max_calls_per_run=3, reference_prefixes=("metric:",)
    ),
    "query_knowledge_rag": ToolSpecification(
        name="query_knowledge_rag", max_calls_per_run=1, reference_prefixes=("knowledge_chunk:",)
    ),
    "search_web": ToolSpecification(
        name="search_web", max_calls_per_run=1, reference_prefixes=("http://", "https://")
    ),
}


def get_tool_specification(tool_name: ToolName) -> ToolSpecification:
    """按稳定名称获取工具约束，避免在调用点散落次数和引用规则。"""
    return TOOL_REGISTRY[tool_name]
