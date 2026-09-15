"""负责调用 DeepAgent 生成回答，并在模型不可用时返回已验证证据。"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.tools import BaseTool

from app.agent.prompt_builder import BASE_SYSTEM_PROMPT, PromptBuilder
from app.agent.review_agent import build_general_subagent, build_review_subagent
from app.config import Settings

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """隔离模型调用与无模型时的受控证据回退。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate_with_llm(
        self,
        user_input: str,
        orchestration_context: str | None,
        tools: Sequence[BaseTool],
    ) -> str | None:
        """调用受限 DeepAgent；异常由调用方返回同轮已验证证据。"""
        try:
            return await self._deep_agent_answer(user_input, orchestration_context, tools)
        except Exception:
            logger.exception("模型回答生成失败，将仅返回已验证的受控工具证据。")
            return None

    async def _deep_agent_answer(
        self,
        user_input: str,
        orchestration_context: str | None,
        tools: Sequence[BaseTool],
    ) -> str:
        """创建最小权限 DeepAgent 并返回最后一条模型消息。"""
        from deepagents import create_deep_agent
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from langchain_openai import ChatOpenAI

        model_options = {
            "model": self.settings.llm_model,
            "api_key": self.settings.llm_api_key,
            "temperature": 0.2,
        }
        if self.settings.llm_base_url:
            # 仅在用户明确配置时传入可选服务端点。
            model_options["base_url"] = self.settings.llm_base_url
        model = ChatOpenAI(**model_options)
        agent = create_deep_agent(
            model=model,
            tools=list(tools),
            subagents=[build_general_subagent(tools), build_review_subagent(tools)],
            # 覆盖框架默认文件系统中间件，只保留其要求的只读能力。
            middleware=[FilesystemMiddleware(tools=["read_file"])],
            system_prompt=BASE_SYSTEM_PROMPT + PromptBuilder.tool_orchestration_prompt(),
        )
        context_suffix = (
            f"\n\n系统编排要求：\n{orchestration_context}" if orchestration_context else ""
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": f"{user_input}{context_suffix}"}]}
        )
        messages = result.get("messages", [])
        if not messages:
            raise RuntimeError("Agent did not return a message")
        content = messages[-1].content
        return content if isinstance(content, str) else str(content)

    @staticmethod
    def generate_evidence_response(data_context: str | None, model_error: bool = False) -> str:
        """模型未配置或失败时，仅返回本轮已验证的工具结果，不再次判断意图。"""
        reason = "模型服务暂时不可用" if model_error else "尚未配置 LLM 服务"
        if not data_context:
            return f"{reason}，且本轮未检索到可用的受控数据、资料或公开网页依据，因此无法生成回答。"
        return (
            f"{reason}，无法生成归因、建议或内容草稿。以下仅展示本轮已验证的检索依据：\n\n"
            f"{data_context}"
        )
