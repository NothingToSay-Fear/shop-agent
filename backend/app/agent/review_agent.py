"""运营复盘子 Agent 的受限职责定义。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.tools import BaseTool
from app.config import Settings

logger = logging.getLogger(__name__)

REVIEW_AGENT_NAME = "operation_review_agent"
GENERAL_PURPOSE_AGENT_NAME = "general-purpose"

REVIEW_AGENT_PROMPT = """你是电商运营复盘专家，只处理活动效果、经营归因、复盘与优化建议。
你不能调用工具、读取文件或补充外部事实；只能消费主 Agent 已验证的数据与资料证据。
对于“GMV 为什么变化”一类问题，必须先使用证据包内已查询的 GMV、支付订单数、访客数、支付转化率、客单价说明可观察到的驱动因素；不能仅因缺少商品或渠道明细，就跳过这些已可验证的指标拆解。仅当需要继续定位渠道、商品/SKU、人群、优惠或活动规则时，才将其写为待验证项，且不得把它们表述为已确认根因。
将“数据事实”“资料依据”“待验证假设”“建议动作”明确区分。
资料或数据未命中时如实说明，不得自行补全。输出简洁的复盘结论，供主 Agent 直接整合。"""


class ReviewAgent:
    """由主工作流显式调用的、无工具证据消费子 Agent。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def review(self, question: str, evidence_package: str) -> str | None:
        if not self.settings.llm_enabled:
            return None
        try:
            from langchain_openai import ChatOpenAI

            options = {"model": self.settings.llm_model, "api_key": self.settings.llm_api_key, "temperature": 0.2}
            if self.settings.llm_base_url:
                options["base_url"] = self.settings.llm_base_url
            model = ChatOpenAI(**options)
            response = await model.ainvoke([
                ("system", REVIEW_AGENT_PROMPT),
                ("user", f"复盘任务：\n{question}\n\n已验证证据包：\n{evidence_package}"),
            ])
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception:
            return None

    async def stream_review(self, question: str, evidence_package: str) -> AsyncIterator[str]:
        """流式输出复盘结论，仍只消费主工作流已验证的证据包。"""
        if not self.settings.llm_enabled:
            return
        try:
            from langchain_openai import ChatOpenAI

            options = {
                "model": self.settings.llm_model,
                "api_key": self.settings.llm_api_key,
                "temperature": 0.2,
            }
            if self.settings.llm_base_url:
                options["base_url"] = self.settings.llm_base_url
            model = ChatOpenAI(**options)
            async for chunk in model.astream([
                ("system", REVIEW_AGENT_PROMPT),
                ("user", f"复盘任务：\n{question}\n\n已验证证据包：\n{evidence_package}"),
            ]):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content
        except Exception:
            logger.exception("复盘子 Agent 流式生成失败")


def build_review_subagent(tools: Sequence[BaseTool]) -> dict[str, object]:
    """构造 DeepAgent 可委派的复盘子 Agent，仅授予受控 RAG 与最小只读文件能力。"""
    return {
        "name": REVIEW_AGENT_NAME,
        "description": "处理活动复盘、经营归因、效果评估和优化建议；会查询受控指标、知识库和必要的公开网络资料。",
        "system_prompt": REVIEW_AGENT_PROMPT,
        # 复盘子 Agent 只消费主 Agent 已验证的证据，不能重复查询或读写文件。
        "tools": [],
    }


def build_general_subagent(tools: Sequence[BaseTool]) -> dict[str, object]:
    """覆盖 DeepAgent 自动附加的通用子 Agent，避免它获得默认的读写和执行能力。"""
    return {
        "name": GENERAL_PURPOSE_AGENT_NAME,
        "description": "处理非复盘类的复杂运营问题；只能使用受控 RAG 工具，不执行文件或外部系统操作。",
        "system_prompt": """你协助主 Agent 处理复杂运营问题。只能依据 query_metric_rag、query_knowledge_rag 和 search_web 的结果回答。
不执行文件写入、命令执行或任何外部操作；资料不足时明确说明。""",
        "tools": list(tools),
        "middleware": [FilesystemMiddleware(tools=["read_file"])],
    }
