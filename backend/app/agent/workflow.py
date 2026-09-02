"""编排意图路由、受控工具调用和最终回答生成。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.agent.answer_generator import AnswerGenerator
from app.agent.execution_plan import ExecutionPlan, build_execution_plan, validate_execution_plan
from app.agent.prompt_builder import PromptBuilder
from app.agent.tools import AgentToolTracker, build_agent_tools
from app.config import Settings
from app.services.conversation_context import (
    ConversationContextSnapshot,
    build_retrieval_question,
)
from app.services.intent_router import RetrievalMode, RetrievalRoute, route_question

StatusCallback = Callable[[str, str], Awaitable[None]]


@dataclass(frozen=True)
class WorkflowResult:
    """一次回答及其实际使用的来源摘要。"""

    answer: str
    data_references: str
    tracker: AgentToolTracker


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
        on_status: StatusCallback | None = None,
        conversation_context: ConversationContextSnapshot | None = None,
    ) -> WorkflowResult:
        """先完成并校验执行计划，再允许模型基于受控结果组织回答。"""
        tracker = AgentToolTracker()
        tools = build_agent_tools(tracker, self.settings)
        active_context = conversation_context or ConversationContextSnapshot()
        retrieval_question = build_retrieval_question(user_input, active_context)
        effective_knowledge_group = knowledge_group or active_context.knowledge_group
        if active_context.display:
            await self._emit_status(on_status, "context", "正在应用本会话已确认的查询条件…")
        await self._emit_status(on_status, "routing", "正在判断问题类型…")
        route = await self._resolve_route(retrieval_question, retrieval_mode)
        tracker.set_route(route)
        plan = build_execution_plan(route)
        await self._emit_status(on_status, "plan", f"已生成执行计划：{plan.summary}")
        await self._run_execution_plan(
            tools,
            retrieval_question,
            effective_knowledge_group,
            plan,
            tracker,
            on_status,
            active_context,
        )
        await self._emit_status(on_status, "verification", "正在校验检索依据…")
        validation = validate_execution_plan(plan, tracker.tool_calls)
        if not validation.passed:
            await self._emit_status(on_status, "verification", "执行计划校验未通过，已停止事实性回答。")
            return WorkflowResult(
                self._verification_failure_answer(validation.summary), tracker.references, tracker
            )

        if self.settings.llm_enabled:
            await self._emit_status(on_status, "generation", "正在基于已验证依据生成结论…")
            answer = await self.answer_generator.generate_with_llm(
                user_input,
                PromptBuilder.execution_instruction(
                    plan,
                    effective_knowledge_group,
                    tracker.data_context,
                    active_context.generation_context,
                ),
                tools,
                effective_knowledge_group,
            )
            if answer is not None:
                return WorkflowResult(answer, tracker.references, tracker)

            await self._emit_status(on_status, "generation", "模型暂不可用，正在生成降级回答…")
            return WorkflowResult(
                self.answer_generator.generate_demo(user_input, tracker.data_context, model_error=True), tracker.references, tracker
            )

        await self._emit_status(on_status, "generation", "正在整理已检索到的依据…")
        return WorkflowResult(
            self.answer_generator.generate_demo(user_input, tracker.data_context), tracker.references, tracker
        )

    async def answer_from_context(
        self, user_input: str, data_context: str | None
    ) -> WorkflowResult:
        """兼容测试和旧调用方的直接上下文路径。"""
        answer = await self.answer_generator.generate_from_context(user_input, data_context)
        return WorkflowResult(answer, "外部注入上下文（仅用于兼容调用）", AgentToolTracker())

    async def _resolve_route(
        self, user_input: str, retrieval_mode: RetrievalMode
    ) -> RetrievalRoute:
        """显式模式优先；默认模式由本地语义路由选择需要的工具。"""
        if retrieval_mode != "hybrid":
            return RetrievalRoute(retrieval_mode, None, 1.0, False)
        return await route_question(user_input, self.settings)

    @staticmethod
    async def _run_execution_plan(
        tools: list[BaseTool],
        user_input: str,
        knowledge_group: str | None,
        plan: ExecutionPlan,
        tracker: AgentToolTracker,
        on_status: StatusCallback | None,
        conversation_context: ConversationContextSnapshot,
    ) -> None:
        """严格按计划执行必调工具；单个工具失败时继续收集其他来源。"""
        tool_by_name = {item.name: item for item in tools}
        for tool_name in plan.required_tools:
            tool = tool_by_name.get(tool_name)
            if tool is None:
                continue
            await AgentWorkflow._emit_status(on_status, "tool", AgentWorkflow._tool_start_message(tool_name))
            payload = {"question": user_input}
            if tool_name == "query_metric_rag":
                payload["start_date"] = getattr(conversation_context, "start_date", None)
                payload["end_date"] = getattr(conversation_context, "end_date", None)
            if tool_name == "query_knowledge_rag":
                payload["group_name"] = knowledge_group
            try:
                await tool.ainvoke(payload)
            except Exception:
                # 各工具包装器已记录稳定错误码；其余资料来源仍应继续尝试。
                pass
            await AgentWorkflow._emit_status(
                on_status, "tool", AgentWorkflow._tool_finish_message(tool_name, tracker)
            )

    @staticmethod
    async def _emit_status(
        on_status: StatusCallback | None, phase: str, content: str
    ) -> None:
        """仅在调用方需要时推送不含用户原文和工具原文的执行进度。"""
        if on_status is not None:
            await on_status(phase, content)

    @staticmethod
    def _tool_start_message(tool_name: str) -> str:
        """为前端展示提供稳定、易理解的工具阶段文案。"""
        return {
            "query_metric_rag": "正在查询经营指标…",
            "query_knowledge_rag": "正在检索知识库资料…",
            "search_web": "正在检索公开互联网资料…",
        }[tool_name]

    @staticmethod
    def _tool_finish_message(tool_name: str, tracker: AgentToolTracker) -> str:
        """依据工具刚写入的审计状态生成简短进度，不暴露完整检索内容。"""
        latest_call = next(
            (item for item in reversed(tracker.tool_calls) if item.tool_name == tool_name), None
        )
        if latest_call is None:
            return "该能力未返回执行记录。"
        labels = {
            "query_metric_rag": "经营指标",
            "query_knowledge_rag": "知识库资料",
            "search_web": "公开互联网资料",
        }
        state_labels = {
            "success": "查询完成",
            "empty": "未检索到可用结果",
            "skipped": "已跳过",
            "failed": "查询失败",
        }
        return f"{labels[tool_name]}{state_labels.get(latest_call.status, '处理完成')}。"

    @staticmethod
    def _verification_failure_answer(validation_summary: str) -> str:
        """计划或引用校验异常时停止事实性回答，避免使用不完整的受控上下文。"""
        return (
            "本轮受控检索未通过完整性校验，暂不基于不完整依据生成结论。"
            f"请稍后重试；校验摘要：{validation_summary}。"
        )
