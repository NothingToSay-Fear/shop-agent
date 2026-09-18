"""负责调用 LangChain 聊天模型生成回答，并在模型不可用时返回已验证证据。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from app.agent.prompt_builder import BASE_SYSTEM_PROMPT
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
    ) -> str | None:
        """调用无工具的 LangChain 聊天模型；异常由调用方返回同轮已验证证据。"""
        try:
            return await self._chat_model_answer(user_input, orchestration_context)
        except Exception:
            logger.exception("模型回答生成失败，将仅返回已验证的受控工具证据。")
            return None

    async def stream_with_llm(
        self,
        user_input: str,
        orchestration_context: str | None,
    ) -> AsyncIterator[str]:
        """逐个转发模型生成的文本片段。

        工具已由主工作流在进入生成层前执行，此处只流式输出最终表达，
        不赋予模型额外的事实获取或工具调用能力。
        """
        try:
            model = self._build_chat_model()
            async for chunk in model.astream(self._messages(user_input, orchestration_context)):
                text = self._stream_chunk_text(chunk)
                if text:
                    yield text
        except Exception:
            logger.exception("模型流式回答生成失败，将返回已验证的受控工具证据。")

    async def _chat_model_answer(
        self,
        user_input: str,
        orchestration_context: str | None,
    ) -> str:
        """调用无工具聊天模型并返回完整回答。"""
        response = await self._build_chat_model().ainvoke(self._messages(user_input, orchestration_context))
        content = response.content
        return content if isinstance(content, str) else str(content)

    def _build_chat_model(self):
        """创建生成用的 LangChain 聊天模型，供完整回答与流式回答共用。"""
        return LLMProviderFactory.create(self.settings, temperature=0.2)

    @staticmethod
    def _messages(user_input: str, orchestration_context: str | None) -> list[tuple[str, str]]:
        context_suffix = f"\n\n系统编排要求：\n{orchestration_context}" if orchestration_context else ""
        return [
            ("system", BASE_SYSTEM_PROMPT),
            ("user", f"{user_input}{context_suffix}"),
        ]

    @staticmethod
    def _stream_chunk_text(chunk: object) -> str:
        """只转发 LangChain 聊天模型的 AI 文本片段。"""
        from langchain_core.messages import AIMessageChunk

        if not isinstance(chunk, AIMessageChunk):
            return ""
        content = chunk.content
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
            f"{reason}，无法生成归因、建议或内容草案。以下仅展示本轮已验证的检索依据：\n\n"
            f"{data_context}"
        )
