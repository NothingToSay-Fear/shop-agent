"""对外提供包含执行阶段的流式运营问答入口。"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from app.agent.streaming import split_answer_fragments
from app.agent.tools import AgentToolTracker
from app.agent.workflow import AgentWorkflow
from app.config import Settings, get_settings
from app.services.conversation_context import ConversationContextSnapshot
from app.services.intent_router import RetrievalMode
from app.services.user_memory import UserMemoryContext


@dataclass(frozen=True)
class AgentStreamEvent:
    """SSE 层可直接映射的进度或文本分片事件。"""

    event_type: Literal["status", "chunk"]
    content: str
    phase: str | None = None


class OperationAgent:
    """仅协调工作流结果与 SSE 文本分片，不承担路由、工具或回答生成细节。"""

    def __init__(self, settings: Settings | None = None) -> None:
        active_settings = settings or get_settings()
        self.workflow = AgentWorkflow(active_settings)
        self.data_references = "演示模式：尚未检索到相关数据、资料或公开网页来源"
        self.tool_tracker = AgentToolTracker()

    async def stream(
        self,
        user_input: str,
        data_context: str | None = None,
        knowledge_group: str | None = None,
        retrieval_mode: RetrievalMode = "hybrid",
        conversation_context: ConversationContextSnapshot | None = None,
        user_memory_context: UserMemoryContext | None = None,
    ) -> AsyncIterator[str]:
        """保留旧文本分片接口，供既有调用方和测试继续使用。"""
        async for event in self.stream_events(
            user_input,
            data_context,
            knowledge_group,
            retrieval_mode,
            conversation_context,
            user_memory_context,
        ):
            if event.event_type == "chunk":
                yield event.content

    async def stream_events(
        self,
        user_input: str,
        data_context: str | None = None,
        knowledge_group: str | None = None,
        retrieval_mode: RetrievalMode = "hybrid",
        conversation_context: ConversationContextSnapshot | None = None,
        user_memory_context: UserMemoryContext | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        """并发接收工作流阶段事件，完成后再流式输出回答文本。"""
        queue: asyncio.Queue[AgentStreamEvent | Exception | object] = asyncio.Queue()
        completed = object()

        async def publish_status(phase: str, content: str) -> None:
            await queue.put(AgentStreamEvent("status", content, phase))

        async def execute() -> None:
            try:
                if data_context is not None:
                    await publish_status("generation", "正在整理已提供的上下文…")
                    result = await self.workflow.answer_from_context(user_input, data_context)
                else:
                    result = await self.workflow.answer(
                        user_input,
                        knowledge_group,
                        retrieval_mode,
                        publish_status,
                        conversation_context,
                        user_memory_context,
                    )
                await queue.put(result)
            except Exception as error:
                await queue.put(error)
            finally:
                await queue.put(completed)

        task = asyncio.create_task(execute(), name="agent-workflow")
        try:
            while True:
                item = await queue.get()
                if item is completed:
                    return
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, AgentStreamEvent):
                    yield item
                    continue
                self.data_references = item.data_references
                self.tool_tracker = item.tracker
                for fragment in split_answer_fragments(item.answer):
                    yield AgentStreamEvent("chunk", fragment)
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
