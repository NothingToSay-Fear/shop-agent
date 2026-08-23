"""运营复盘子 Agent 的受限职责定义。"""

from __future__ import annotations

from collections.abc import Sequence

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.tools import BaseTool

REVIEW_AGENT_NAME = "operation_review_agent"
GENERAL_PURPOSE_AGENT_NAME = "general-purpose"

REVIEW_AGENT_PROMPT = """你是电商运营复盘专家，只处理活动效果、经营归因、复盘与优化建议。
先调用 query_metric_rag 获取当前或指定周期的数据；问题涉及活动规则、历史方案或复盘资料时，必须再调用 query_knowledge_rag。
只能根据工具返回的数值和资料陈述事实；将“数据事实”“资料依据”“待验证假设”“建议动作”明确区分。
资料或数据未命中时如实说明，不得自行补全。输出简洁的复盘结论，供主 Agent 直接整合。"""


def build_review_subagent(tools: Sequence[BaseTool]) -> dict[str, object]:
    """构造 DeepAgent 可委派的复盘子 Agent，仅授予受控 RAG 与最小只读文件能力。"""
    return {
        "name": REVIEW_AGENT_NAME,
        "description": "处理活动复盘、经营归因、效果评估和优化建议；会查询受控指标与知识库依据。",
        "system_prompt": REVIEW_AGENT_PROMPT,
        "tools": list(tools),
        # DeepAgent 的文件系统中间件要求至少保留 read_file；不开放写入或命令执行。
        "middleware": [FilesystemMiddleware(tools=["read_file"])],
    }


def build_general_subagent(tools: Sequence[BaseTool]) -> dict[str, object]:
    """覆盖 DeepAgent 自动附加的通用子 Agent，避免它获得默认的读写和执行能力。"""
    return {
        "name": GENERAL_PURPOSE_AGENT_NAME,
        "description": "处理非复盘类的复杂运营问题；只能使用受控 RAG 工具，不执行文件或外部系统操作。",
        "system_prompt": """你协助主 Agent 处理复杂运营问题。只能依据 query_metric_rag 和 query_knowledge_rag 的结果回答。
不执行文件写入、命令执行或任何外部操作；资料不足时明确说明。""",
        "tools": list(tools),
        "middleware": [FilesystemMiddleware(tools=["read_file"])],
    }
