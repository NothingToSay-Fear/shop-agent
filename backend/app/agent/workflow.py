"""编排意图路由、受控工具调用和最终回答生成。"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.agent.answer_generator import AnswerGenerator
from app.agent.prompt_builder import PromptBuilder
from app.agent.tools import AgentToolTracker, build_agent_tools
from app.config import Settings
from app.services.intent_router import RetrievalMode, RetrievalRoute, route_question


@dataclass(frozen=True)
class WorkflowResult:
    """一次回答及其实际使用的来源摘要。"""

    answer: str
    data_references: str


class AgentWorkflow:
    """集中处理路由与工具执行，不包含 SSE 输出或具体回答文案。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.answer_generator = AnswerGenerator(settings)

    async def answer(
        self,
        user_input: str,
        knowledge_group: str | None,
        retrieval_mode: RetrievalMode,
    ) -> WorkflowResult:
        """按路由调用工具；模型失败时以同一批工具结果进入演示回答。"""
        tracker = AgentToolTracker()
        tools = build_agent_tools(tracker, self.settings)
        route = await self._resolve_route(user_input, retrieval_mode)
        if self.settings.llm_enabled:
            answer = await self.answer_generator.generate_with_llm(
                user_input,
                PromptBuilder.routing_instruction(route, knowledge_group),
                tools,
                knowledge_group,
            )
            if answer is not None:
                return WorkflowResult(answer, tracker.references)

            await self._run_tools_for_route(tools, user_input, knowledge_group, route)
            return WorkflowResult(
                self.answer_generator.generate_demo(user_input, tracker.data_context, model_error=True),
                tracker.references,
            )

        await self._run_tools_for_route(tools, user_input, knowledge_group, route)
        return WorkflowResult(
            self.answer_generator.generate_demo(user_input, tracker.data_context), tracker.references
        )

    async def answer_from_context(
        self, user_input: str, data_context: str | None
    ) -> WorkflowResult:
        """兼容测试和旧调用方的直接上下文路径。"""
        answer = await self.answer_generator.generate_from_context(user_input, data_context)
        return WorkflowResult(answer, "外部注入上下文（仅用于兼容调用）")

    async def _resolve_route(
        self, user_input: str, retrieval_mode: RetrievalMode
    ) -> RetrievalRoute:
        """显式模式优先；默认模式由本地语义路由选择需要的工具。"""
        if retrieval_mode != "hybrid":
            return RetrievalRoute(retrieval_mode, None, 1.0, False)
        return await route_question(user_input, self.settings)

    @staticmethod
    async def _run_tools_for_route(
        tools: list[BaseTool],
        user_input: str,
        knowledge_group: str | None,
        route: RetrievalRoute,
    ) -> None:
        """严格按路由调用工具，避免 API 层重新编排检索。"""
        tool_by_name = {item.name: item for item in tools}
        if route.mode in ("metrics", "hybrid", "web_hybrid"):
            await tool_by_name["query_metric_rag"].ainvoke({"question": user_input})
        if route.mode in ("knowledge", "hybrid", "web_hybrid"):
            await tool_by_name["query_knowledge_rag"].ainvoke(
                {"question": user_input, "group_name": knowledge_group}
            )
        if route.mode in ("web", "web_hybrid"):
            await tool_by_name["search_web"].ainvoke({"question": user_input})
