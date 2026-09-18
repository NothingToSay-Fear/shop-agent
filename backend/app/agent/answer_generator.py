"""负责调用 DeepAgent 生成回答，并在模型不可用时返回已验证证据。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence

from langchain_core.tools import BaseTool

from app.agent.prompt_builder import BASE_SYSTEM_PROMPT, PromptBuilder
from app.config import Settings
from app.services.llm_factory import LLMProviderFactory

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
        use_review_agent: bool = False,
    ) -> str | None:
        """调用受限 DeepAgent；异常由调用方返回同轮已验证证据。"""
        try:
            return await self._deep_agent_answer(user_input, orchestration_context, tools, use_review_agent)
        except Exception:
            logger.exception("模型回答生成失败，将仅返回已验证的受控工具证据。")
            return None

    async def stream_with_llm(
        self,
        user_input: str,
        orchestration_context: str | None,
        tools: Sequence[BaseTool],
    ) -> AsyncIterator[str]:
        """逐个转发模型生成的文本片段。

工具已由主工作流在进入生成层前执行，此处只流式输出最终表达，不赋予模型额外的事实获取能力。
        """
        try:
            agent = self._build_deep_agent(tools)
            context_suffix = (
                f"\n\n系统编排要求：\n{orchestration_context}"
                if orchestration_context
                else ""
            )
            async for event in agent.astream(
                {"messages": [{"role": "user", "content": f"{user_input}{context_suffix}"}]},
                stream_mode="messages",
            ):
                text = self._stream_event_text(event)
                if text:
                    yield text
        except Exception:
            logger.exception("模型流式回答生成失败，将返回已验证的受控工具证据。")

    async def _deep_agent_answer(
        self,
        user_input: str,
        orchestration_context: str | None,
        tools: Sequence[BaseTool],
        use_review_agent: bool,
    ) -> str:
        """创建最小权限 DeepAgent 并返回最后一条模型消息。"""
        agent = self._build_deep_agent(tools)
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

    def _build_deep_agent(self, tools: Sequence[BaseTool]):
        """创建生成用的最小权限 DeepAgent，供同步完整回答与流式回答共用。"""
        from deepagents import create_deep_agent
        from deepagents.middleware.filesystem import FilesystemMiddleware
        return create_deep_agent(
            model=LLMProviderFactory.create(self.settings, temperature=0.2),
            tools=list(tools),
            subagents=[],
            middleware=[FilesystemMiddleware(tools=["read_file"])],
            system_prompt=BASE_SYSTEM_PROMPT + PromptBuilder.tool_orchestration_prompt(),
        )

    @staticmethod
    def _stream_event_text(event: object) -> str:
        """只转发最终 AI 消息片段，忽略 LangGraph 的元数据和工具消息。"""
        from langchain_core.messages import AIMessageChunk

        if not isinstance(event, tuple) or not event:
            return ""
        message = event[0]
        if not isinstance(message, AIMessageChunk):
            return ""
        content = message.content
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )

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
