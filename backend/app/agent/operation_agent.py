"""对外提供流式运营问答入口。"""

from collections.abc import AsyncIterator

from app.agent.streaming import split_answer_fragments
from app.agent.tools import AgentToolTracker
from app.agent.workflow import AgentWorkflow
from app.config import Settings, get_settings
from app.services.intent_router import RetrievalMode


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
    ) -> AsyncIterator[str]:
        """执行工作流并将最终回答切分为适合 SSE 推送的片段。"""
        if data_context is not None:
            result = await self.workflow.answer_from_context(user_input, data_context)
        else:
            result = await self.workflow.answer(user_input, knowledge_group, retrieval_mode)
        self.data_references = result.data_references
        self.tool_tracker = result.tracker
        for fragment in split_answer_fragments(result.answer):
            yield fragment
