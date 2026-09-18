"""编排意图路由、受控工具调用和最终回答生成。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.agent.answer_generator import AnswerGenerator
from app.agent.data_query_agent import DataQueryAgent
from app.agent.review_agent import ReviewAgent
from app.agent.task_orchestrator import DelegationPlan, MainAgentOrchestrator
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
from app.services.task_plans import ExecutablePlanAction, PlanExecutionController

StatusCallback = Callable[[str, str], Awaitable[None]]
ChunkCallback = Callable[[str], Awaitable[None]]


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
        on_chunk: ChunkCallback | None = None,
        plan_actions: list[ExecutablePlanAction] | None = None,
        plan_controller: PlanExecutionController | None = None,
    ) -> WorkflowResult:
        """先完成并校验执行计划，再允许模型基于受控结果组织回答。"""
        tracker = AgentToolTracker(run_id=plan_controller.run_id if plan_controller else None)
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
        pending_final_actions: dict[str, ExecutablePlanAction] = {}
        if plan_actions is None:
            await self._run_execution_plan(
                tools,
                retrieval_question,
                plan,
                tracker,
                on_status,
                active_context,
                data_query_agent,
            )
        else:
            pending_final_actions = await self._run_persisted_actions(
                tools,
                retrieval_question,
                tracker,
                on_status,
                active_context,
                data_query_agent,
                delegation_plan,
                plan_actions,
                plan_controller,
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
            review_action = pending_final_actions.get("review")
            review_attempt = await self._begin_action(plan_controller, review_action)
            reviewed_answer = await self._generate_review_answer(
                user_input, tracker.data_context or "本轮没有可用证据。", on_chunk
            )
            if reviewed_answer is not None:
                await self._finish_action(
                    plan_controller, review_action, review_attempt, "success", "已基于验证证据生成复盘结论", tracker.reference_ids
                )
                synth_action = pending_final_actions.get("synthesize")
                synth_attempt = await self._begin_action(plan_controller, synth_action)
                await self._finish_action(
                    plan_controller, synth_action, synth_attempt, "success", "已形成最终受控结论", tracker.reference_ids
                )
                return WorkflowResult(reviewed_answer, tracker.references, tracker)
            await self._finish_action(
                plan_controller, review_action, review_attempt, "failed", "复盘子 Agent 未返回结论", (), "review_agent_unavailable"
            )

        if self.settings.llm_enabled:
            await self._emit_status(on_status, "generation", "正在基于已验证依据生成结论…")
            synth_action = pending_final_actions.get("synthesize")
            synth_attempt = await self._begin_action(plan_controller, synth_action)
            generation_context = PromptBuilder.execution_instruction(
                plan,
                tracker.data_context,
                active_context.generation_context,
                active_memory_context.display,
                active_summary_context.display,
                active_history_context.display,
                delegation_plan.output_contract if delegation_plan else "",
            )
            answer = await self._generate_answer(user_input, generation_context, tools, on_chunk)
            if answer is not None:
                await self._finish_action(
                    plan_controller, synth_action, synth_attempt, "success", "已形成最终受控结论", tracker.reference_ids
                )
                return WorkflowResult(answer, tracker.references, tracker)

            await self._emit_status(on_status, "generation", "模型暂不可用，正在返回已验证依据…")
            answer = self.answer_generator.generate_evidence_response(
                tracker.data_context, model_error=True
            )
            await self._emit_chunk(on_chunk, answer)
            await self._finish_action(
                plan_controller, synth_action, synth_attempt, "completed", "模型不可用，已返回验证证据", tracker.reference_ids
            )
            return WorkflowResult(answer, tracker.references, tracker)

        await self._emit_status(on_status, "generation", "LLM 未配置，正在返回已验证依据…")
        answer = self.answer_generator.generate_evidence_response(tracker.data_context)
        await self._emit_chunk(on_chunk, answer)
        synth_action = pending_final_actions.get("synthesize")
        synth_attempt = await self._begin_action(plan_controller, synth_action)
        await self._finish_action(
            plan_controller, synth_action, synth_attempt, "completed", "LLM 未配置，已返回验证证据", tracker.reference_ids
        )
        return WorkflowResult(answer, tracker.references, tracker)

    async def _generate_answer(
        self,
        user_input: str,
        generation_context: str,
        tools: list[BaseTool],
        on_chunk: ChunkCallback | None,
    ) -> str | None:
        """通常请求逐片转发模型生成。没有 SSE 消费者时保留完整回答接口，供同步调用和旧测试使用。"""
        if on_chunk is None:
            return await self.answer_generator.generate_with_llm(user_input, generation_context, tools)

        parts: list[str] = []
        async for chunk in self.answer_generator.stream_with_llm(user_input, generation_context, tools):
            parts.append(chunk)
            await self._emit_chunk(on_chunk, chunk)
        return "".join(parts) or None

    async def _generate_review_answer(
        self,
        user_input: str,
        evidence_package: str,
        on_chunk: ChunkCallback | None,
    ) -> str | None:
        review_agent = ReviewAgent(self.settings)
        if on_chunk is None:
            return await review_agent.review(user_input, evidence_package)

        parts: list[str] = []
        async for chunk in review_agent.stream_review(user_input, evidence_package):
            parts.append(chunk)
            await self._emit_chunk(on_chunk, chunk)
        return "".join(parts) or None

    async def _resolve_route(
        self,
        user_input: str,
        retrieval_mode: RetrievalMode,
    ) -> RetrievalRoute:
        """显式模式优先；默认模式由本地语义路由选择需要的工具。"""
        if retrieval_mode != "hybrid":
            return RetrievalRoute(retrieval_mode, None, 1.0, False)
        return await route_question(user_input, self.settings)

    async def _run_persisted_actions(
        self,
        tools: list[BaseTool],
        user_input: str,
        tracker: AgentToolTracker,
        on_status: StatusCallback | None,
        conversation_context: ConversationContextSnapshot,
        data_query_agent: DataQueryAgent | None,
        delegation_plan: DelegationPlan | None,
        actions: list[ExecutablePlanAction],
        controller: PlanExecutionController | None,
    ) -> dict[str, ExecutablePlanAction]:
        """执行数据库中的 Action 队列，并仅在观察结果允许时追加下一版计划。"""
        tool_by_name = {item.name: item for item in tools}
        completed_keys: set[str] = set()
        deferred: dict[str, ExecutablePlanAction] = {}
        queue = list(actions)
        replan_count = 0

        while queue:
            action = queue.pop(0)
            if action.action_type in {"review", "synthesize"}:
                deferred["review" if action.action_type == "review" else "synthesize"] = action
                continue
            if any(dependency not in completed_keys for dependency in action.depends_on):
                attempt = await self._begin_action(controller, action)
                await self._finish_action(
                    controller,
                    action,
                    attempt,
                    "skipped",
                    "前置动作未完成，跳过本动作",
                )
                completed_keys.add(action.key)
                continue

            attempt = await self._begin_action(controller, action)
            if action.action_type == "confirm_constraints":
                await self._finish_action(
                    controller,
                    action,
                    attempt,
                    "completed",
                    "已使用任务有效约束和能力目录生成执行范围",
                )
                completed_keys.add(action.key)
                continue

            if action.action_type in {"query_metrics", "query_knowledge"}:
                tool_name = action.tool_name
                tool = tool_by_name.get(tool_name or "")
                max_tool_calls = int((controller.plan.budget or {}).get("max_tool_calls", 0)) if controller else 0
                if max_tool_calls and len(tracker.tool_calls) >= max_tool_calls:
                    await self._finish_action(
                        controller, action, attempt, "skipped", "已达到本计划的工具调用预算", (), "tool_budget_exhausted"
                    )
                    completed_keys.add(action.key)
                    continue
                if tool is None:
                    await self._finish_action(
                        controller, action, attempt, "failed", "计划工具未注册", (), "tool_not_registered"
                    )
                    completed_keys.add(action.key)
                    continue
                await self._emit_status(on_status, "tool", self._tool_start_message(tool_name))
                call_offset = len(tracker.tool_calls)
                try:
                    if action.action_type == "query_metrics" and data_query_agent is not None:
                        await self._emit_status(
                            on_status, "data_query_plan", "数据库子 Agent 正在确认指标、时间范围与数据能力…"
                        )
                        raw_codes = action.action_input.get("metric_codes")
                        metric_codes = (
                            tuple(item for item in raw_codes if isinstance(item, str))
                            if isinstance(raw_codes, list)
                            else None
                        )
                        await data_query_agent.execute(
                            tool, user_input, conversation_context, metric_codes=metric_codes
                        )
                    else:
                        await tool.ainvoke({"question": user_input})
                except Exception:
                    pass
                latest = next(
                    (item for item in reversed(tracker.tool_calls[call_offset:]) if item.tool_name == tool_name),
                    None,
                )
                if latest is None:
                    await self._finish_action(
                        controller, action, attempt, "failed", "工具未返回可审计结果", (), "missing_tool_audit"
                    )
                else:
                    await self._finish_action(
                        controller,
                        action,
                        attempt,
                        latest.status,
                        latest.result_summary,
                        latest.reference_ids,
                        latest.error_code,
                    )
                await self._emit_status(on_status, "tool", self._tool_finish_message(tool_name, tracker))
                completed_keys.add(action.key)
                continue

            if action.action_type == "evaluate_evidence":
                observation = data_query_agent.observe(tracker) if data_query_agent is not None else None
                if observation is None:
                    summary = "未配置数据库子 Agent，无法提出下一步数据动作"
                elif observation.blocked_by_capability:
                    summary = observation.reason + "；" + "；".join(observation.blocked_by_capability)
                else:
                    summary = observation.reason
                await self._finish_action(controller, action, attempt, "completed", summary, tracker.reference_ids)
                completed_keys.add(action.key)
                if (
                    observation is not None
                    and observation.next_metric_codes
                    and delegation_plan is not None
                    and controller is not None
                ):
                    proposed = MainAgentOrchestrator().replan(
                        delegation_plan,
                        next_metric_codes=observation.next_metric_codes,
                        reason=observation.reason,
                    )
                    if proposed:
                        await self._emit_status(on_status, "replan", "主 Agent 已根据证据追加下一批数据验证动作…")
                        pending = await controller.append_replan(proposed, observation.reason)
                        replan_count += 1
                        # 已延后的最终动作将由最后一版计划重新读取；只追加新的调查动作。
                        queue = [
                            item
                            for item in pending
                            if item.key not in completed_keys
                            and item.action_type not in {"review", "synthesize"}
                        ]
                        for item in pending:
                            if item.action_type in {"review", "synthesize"}:
                                deferred["review" if item.action_type == "review" else "synthesize"] = item
                continue

            await self._finish_action(
                controller, action, attempt, "failed", "不支持的计划动作", (), "unsupported_plan_action"
            )
            completed_keys.add(action.key)

        return deferred

    @staticmethod
    async def _begin_action(
        controller: PlanExecutionController | None, action: ExecutablePlanAction | None
    ) -> object | None:
        if controller is None or action is None:
            return None
        return await controller.begin(action)

    @staticmethod
    async def _finish_action(
        controller: PlanExecutionController | None,
        action: ExecutablePlanAction | None,
        attempt: object | None,
        status: str,
        result_summary: str,
        evidence_references: tuple[str, ...] = (),
        error_message: str | None = None,
    ) -> None:
        if controller is None or action is None or attempt is None:
            return
        await controller.finish(
            action,
            attempt,  # type: ignore[arg-type]
            status=status,
            result_summary=result_summary,
            evidence_references=evidence_references,
            error_message=error_message,
        )

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
    async def _emit_chunk(on_chunk: ChunkCallback | None, content: str) -> None:
        if on_chunk is not None and content:
            await on_chunk(content)

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
