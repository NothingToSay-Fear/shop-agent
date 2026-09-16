"""编排意图路由、受控工具调用和最终回答生成。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.agent.answer_generator import AnswerGenerator
from app.agent.data_query_agent import DataQueryAgent
from app.agent.review_agent import ReviewAgent
from app.agent.task_orchestrator import DelegationPlan
from app.agent.execution_plan import ExecutionPlan, build_execution_plan, validate_execution_plan
from app.agent.prompt_builder import PromptBuilder
from app.agent.tools import AgentToolTracker, build_agent_tools
from app.config import Settings
from app.services.conversation_context import (
    ConversationContextSnapshot,
    build_retrieval_question,
)
from app.services.conversation_summary import ConversationSummaryContext
from app.services.conversation_history import ConversationHistoryContext
from app.services.intent_router import RetrievalMode, RetrievalRoute, route_question
from app.services.query_expansion import prepare_retrieval_queries
from app.services.user_memory import UserMemoryContext

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
        retrieval_mode: RetrievalMode,
        on_status: StatusCallback | None = None,
        conversation_context: ConversationContextSnapshot | None = None,
        user_memory_context: UserMemoryContext | None = None,
        user_id: str | None = None,
        conversation_summary_context: ConversationSummaryContext | None = None,
        conversation_history_context: ConversationHistoryContext | None = None,
        route_override: RetrievalRoute | None = None,
        data_query_agent: DataQueryAgent | None = None,
        delegation_plan: DelegationPlan | None = None,
    ) -> WorkflowResult:
        """先完成并校验执行计划，再允许模型基于受控结果组织回答。"""
        tracker = AgentToolTracker()
        active_context = conversation_context or ConversationContextSnapshot()
        active_memory_context = user_memory_context or UserMemoryContext()
        active_summary_context = conversation_summary_context or ConversationSummaryContext()
        active_history_context = conversation_history_context or ConversationHistoryContext()
        retrieval_question = build_retrieval_question(user_input, active_context)
        if active_context.display:
            await self._emit_status(on_status, "context", "正在应用本会话已确认的查询条件…")
        if active_memory_context.items:
            await self._emit_status(on_status, "memory", "正在应用你的长期偏好…")
        if active_summary_context.used:
            await self._emit_status(on_status, "summary", "正在恢复本会话短期状态…")
        if active_history_context.used:
            await self._emit_status(on_status, "history", "正在补充相关历史讨论…")
        await self._emit_status(on_status, "routing", "正在判断问题类型…")
        # 本轮意图只由用户原始表达判断；会话条件仅用于限定后续真实检索，避免旧条件放大综合意图。
        route = (
            route_override
            if route_override is not None
            else await self._resolve_route(user_input, retrieval_mode)
        )
        if delegation_plan is not None:
            route = RetrievalRoute(delegation_plan.route_mode, None, 1.0, False, "main_agent_delegation")
        tracker.set_route(route)
        plan = build_execution_plan(route)
        prepared_queries = None
        if any(
            tool_name in {"query_metric_rag", "query_knowledge_rag"}
            for tool_name in plan.required_tools
        ):
            prepared_queries = await prepare_retrieval_queries(
                retrieval_question,
                self.settings,
                route.query_embedding if retrieval_question == user_input else None,
            )
        tools = build_agent_tools(tracker, self.settings, user_id, prepared_queries)
        batch_summary = " → ".join(" + ".join(batch) for batch in delegation_plan.batches) if delegation_plan else plan.summary
        await self._emit_status(on_status, "plan", f"主 Agent 已生成执行批次：{batch_summary}")
        await self._run_execution_plan(
            tools,
            retrieval_question,
            plan,
            tracker,
            on_status,
            active_context,
            data_query_agent,
        )
        await self._emit_status(on_status, "verification", "正在校验检索依据…")
        validation = validate_execution_plan(plan, tracker.tool_calls)
        if not validation.passed:
            await self._emit_status(on_status, "verification", "执行计划校验未通过，已停止事实性回答。")
            return WorkflowResult(
                self._verification_failure_answer(validation.summary), tracker.references, tracker
            )

        if delegation_plan is not None and delegation_plan.use_review_agent:
            await self._emit_status(on_status, "review", "正在将已验证证据包委派给复盘子 Agent…")
            reviewed_answer = await ReviewAgent(self.settings).review(
                user_input, tracker.data_context or "本轮没有可用证据。"
            )
            if reviewed_answer is not None:
                return WorkflowResult(reviewed_answer, tracker.references, tracker)

        if self.settings.llm_enabled:
            await self._emit_status(on_status, "generation", "正在基于已验证依据生成结论…")
            generation_context = PromptBuilder.execution_instruction(
                plan,
                tracker.data_context,
                active_context.generation_context,
                active_memory_context.display,
                active_summary_context.display,
                active_history_context.display,
                delegation_plan.output_contract if delegation_plan else "",
            )
            if delegation_plan is None:
                answer = await self.answer_generator.generate_with_llm(
                    user_input, generation_context, tools
                )
            else:
                answer = await self.answer_generator.generate_with_llm(
                    user_input,
                    generation_context,
                    tools,
                    use_review_agent=False,
                )
            if answer is not None:
                return WorkflowResult(answer, tracker.references, tracker)

            await self._emit_status(on_status, "generation", "模型暂不可用，正在返回已验证依据…")
            return WorkflowResult(
                self.answer_generator.generate_evidence_response(
                    tracker.data_context, model_error=True
                ),
                tracker.references,
                tracker,
            )

        await self._emit_status(on_status, "generation", "LLM 未配置，正在返回已验证依据…")
        return WorkflowResult(
            self.answer_generator.generate_evidence_response(tracker.data_context),
            tracker.references,
            tracker,
        )

    async def _resolve_route(
        self,
        user_input: str,
        retrieval_mode: RetrievalMode,
    ) -> RetrievalRoute:
        """显式模式优先；默认模式由本地语义路由选择需要的工具。"""
        if retrieval_mode != "hybrid":
            return RetrievalRoute(retrieval_mode, None, 1.0, False)
        return await route_question(user_input, self.settings)

    @staticmethod
    async def _run_execution_plan(
        tools: list[BaseTool],
        user_input: str,
        plan: ExecutionPlan,
        tracker: AgentToolTracker,
        on_status: StatusCallback | None,
        conversation_context: ConversationContextSnapshot,
        data_query_agent: DataQueryAgent | None = None,
    ) -> None:
        """严格按计划执行必调工具；单个工具失败时继续收集其他来源。"""
        tool_by_name = {item.name: item for item in tools}
        for tool_name in plan.required_tools:
            tool = tool_by_name.get(tool_name)
            if tool is None:
                continue
            await AgentWorkflow._emit_status(on_status, "tool", AgentWorkflow._tool_start_message(tool_name))
            try:
                if tool_name == "query_metric_rag" and data_query_agent is not None:
                    await AgentWorkflow._emit_status(
                        on_status, "data_query_plan", "数据查询子 Agent 正在确认指标、时间范围与数据能力…"
                    )
                    await data_query_agent.execute(tool, user_input, conversation_context)
                else:
                    payload: dict[str, object] = {"question": user_input}
                    if tool_name == "query_metric_rag":
                        payload["start_date"] = conversation_context.start_date
                        payload["end_date"] = conversation_context.end_date
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
