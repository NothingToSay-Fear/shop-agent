"""Agent entry point with an offline-safe development fallback."""

from collections.abc import AsyncIterator

from app.agent.rag_tools import RagToolTracker, build_rag_tools
from app.agent.review_agent import (
    REVIEW_AGENT_NAME,
    build_general_subagent,
    build_review_subagent,
)
from app.config import Settings, get_settings
from app.services.intent_router import RetrievalMode, RetrievalRoute, route_question

SYSTEM_PROMPT = """你是 Shop Agent，一名电商运营工作助手。
你只能依据用户给出的信息和工具结果陈述事实；不确定时明确说明。
回答使用中文，结构清晰，并给出可执行的下一步与验证指标。
外部平台写操作必须提醒用户人工确认。"""


class OperationAgent:
    """通过 DeepAgent 或离线演示兜底返回流式运营回答。"""
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.data_references = "演示模式：尚未检索到相关数据或资料"

    async def stream(
        self,
        user_input: str,
        data_context: str | None = None,
        knowledge_group: str | None = None,
        retrieval_mode: RetrievalMode = "hybrid",
    ) -> AsyncIterator[str]:
        """执行一次主 Agent 编排；data_context 仅保留给已有离线测试和兼容调用。"""
        if data_context is not None:
            answer = await self._answer_with_context(user_input, data_context)
        else:
            answer = await self._answer_with_rag_tools(user_input, knowledge_group, retrieval_mode)
        for fragment in self._fragments(answer):
            yield fragment

    async def _answer_with_context(self, user_input: str, data_context: str | None) -> str:
        """兼容已注入上下文的调用；生产请求统一经 RAG 工具编排。"""
        if self.settings.llm_enabled:
            try:
                return await self._deep_agent_answer(user_input, data_context, [], None)
            except Exception:
                # 已配置的模型服务暂不可用时，仍保证本地操作可使用。
                return self._demo_answer(user_input, data_context, model_error=True)
        return self._demo_answer(user_input, data_context)

    async def _answer_with_rag_tools(
        self,
        user_input: str,
        knowledge_group: str | None,
        retrieval_mode: RetrievalMode,
    ) -> str:
        """由主 Agent 使用工具和子 Agent；模型不可用时仍以同一工具完成确定性检索。"""
        tracker = RagToolTracker()
        tools = build_rag_tools(tracker, self.settings)
        route = await self._resolve_route(user_input, retrieval_mode)
        if self.settings.llm_enabled:
            try:
                answer = await self._deep_agent_answer(
                    user_input, self._routing_instruction(route, knowledge_group), tools, knowledge_group
                )
                self.data_references = tracker.references
                return answer
            except Exception:
                await self._run_rag_tools_for_route(tools, user_input, knowledge_group, route)
                self.data_references = tracker.references
                return self._demo_answer(user_input, tracker.data_context, model_error=True)

        await self._run_rag_tools_for_route(tools, user_input, knowledge_group, route)
        self.data_references = tracker.references
        return self._demo_answer(user_input, tracker.data_context)

    async def _deep_agent_answer(
        self,
        user_input: str,
        orchestration_context: str | None,
        tools,
        knowledge_group: str | None,
    ) -> str:
        # ChatOpenAI 可通过 LLM_BASE_URL 对接 OpenAI 兼容协议的服务端点。
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
            tools=tools,
            subagents=[build_general_subagent(tools), build_review_subagent(tools)],
            # 覆盖框架默认文件系统中间件，只保留其要求的只读能力。
            middleware=[FilesystemMiddleware(tools=["read_file"])],
            system_prompt=SYSTEM_PROMPT + self._tool_orchestration_prompt(knowledge_group),
        )
        context_suffix = f"\n\n系统编排要求：\n{orchestration_context}" if orchestration_context else ""
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": f"{user_input}{context_suffix}"}]}
        )
        messages = result.get("messages", [])
        if not messages:
            raise RuntimeError("Agent did not return a message")
        content = messages[-1].content
        return content if isinstance(content, str) else str(content)

    async def _resolve_route(
        self, user_input: str, retrieval_mode: RetrievalMode
    ) -> RetrievalRoute:
        """显式模式优先；默认模式由本地语义路由选择需要的工具。"""
        if retrieval_mode != "hybrid":
            return RetrievalRoute(retrieval_mode, None, 1.0, False)
        return await route_question(user_input, self.settings)

    @staticmethod
    async def _run_rag_tools_for_route(
        tools, user_input: str, knowledge_group: str | None, route: RetrievalRoute
    ) -> None:
        """离线兜底严格按路由调用同一批工具，避免 API 层重新编排 RAG。"""
        tool_by_name = {item.name: item for item in tools}
        if route.mode in ("metrics", "hybrid"):
            await tool_by_name["query_metric_rag"].ainvoke({"question": user_input})
        if route.mode in ("knowledge", "hybrid"):
            await tool_by_name["query_knowledge_rag"].ainvoke(
                {"question": user_input, "group_name": knowledge_group}
            )

    @staticmethod
    def _routing_instruction(route: RetrievalRoute, knowledge_group: str | None) -> str:
        """把本地路由结论转成主 Agent 必须遵守的工具调用约束。"""
        group_hint = f"知识库分组限定为“{knowledge_group}”。" if knowledge_group else "未限定知识库分组。"
        if route.mode == "metrics":
            return f"{group_hint} 该问题被判定为指标查询，必须调用 query_metric_rag。"
        if route.mode == "knowledge":
            return f"{group_hint} 该问题被判定为知识库问答，必须调用 query_knowledge_rag。"
        return (
            f"{group_hint} 该问题需要综合分析或路由置信度不足，必须调用 query_metric_rag 和 "
            f"query_knowledge_rag；如涉及复盘、归因、效果评估或优化建议，必须通过 task 委派给 "
            f"{REVIEW_AGENT_NAME}。"
        )

    @staticmethod
    def _tool_orchestration_prompt(knowledge_group: str | None) -> str:
        """明确主 Agent、工具和复盘子 Agent 的职责边界。"""
        group_hint = (
            f"用户限定知识库分组为“{knowledge_group}”，调用知识库工具时必须传入该分组。"
            if knowledge_group
            else "用户未限定知识库分组，可检索全部已就绪资料。"
        )
        return f"""

你可以调用 query_metric_rag 和 query_knowledge_rag 两个受控工具。不得自行编造数据、规则或引用。
系统会在用户消息末尾提供必须遵守的工具调用要求；工具结果是唯一可用于数据事实和资料事实的依据。
当要求中指定复盘子 Agent 时，必须使用 task 委派给 {REVIEW_AGENT_NAME}，再整合其结论。
{group_hint}
"""

    @staticmethod
    def _demo_answer(
        user_input: str, data_context: str | None = None, model_error: bool = False
    ) -> str:
        """提供确定性的本地回答，覆盖主要运营场景。"""
        lower_input = user_input.lower()
        prefix = "当前为演示模式，以下内容未查询真实业务数据。\n\n"
        if data_context:
            prefix = "当前为演示模式，以下结论基于内置模拟经营数据。\n\n"
        if model_error:
            prefix = "模型服务暂不可用，已切换到演示模式。以下内容未查询真实业务数据。\n\n"

        if data_context and "【知识库资料】" in data_context and "【经营指标】" not in data_context:
            if "未检索到" in data_context:
                return "当前知识库中未检索到足以回答该问题的资料，请上传或选择相关的运营活动、玩法规则文件后再试。"
            return (
                "当前为演示模式。以下回答仅整理知识库检索到的资料，不补充资料中未出现的事实。\n\n"
                f"## 检索依据\n{data_context}\n\n"
                "请配置 LLM 服务后，系统会基于上述依据生成更完整的归纳回答。"
            )

        if data_context and "【经营指标】" in data_context and "【知识库资料】" in data_context:
            return (
                "当前为演示模式。以下内容同时包含经营指标与知识库资料，请将数据事实和历史经验分开理解。\n\n"
                f"## 综合依据\n{data_context}\n\n"
                "请配置 LLM 服务后，系统会基于两类依据生成完整的归因、建议与验证动作。"
            )

        if any(keyword in lower_input for keyword in ("gmv", "环比", "同比", "订单", "转化", "客单", "退款", "流量")):
            metric_summary = data_context or "暂无可用经营数据，请先执行模拟数据初始化。"
            return (
                f"{prefix}## 经营数据分析\n"
                f"你关注的是：{user_input}\n\n"
                f"{metric_summary}\n\n"
                "**初步诊断**：当前模拟数据中，访客下降幅度大于转化率改善带来的收益，"
                "应优先检查内容种草与搜索广告的曝光、点击和预算变化。\n"
                "**下一步**：按渠道拆分访客和支付订单，确认流量下降是否集中在单一渠道。\n"
                "**验证指标**：GMV、支付转化率、访客数、客单价。"
            )
        if any(keyword in user_input for keyword in ("标题", "文案", "卖点", "详情页")):
            return (
                f"{prefix}## 内容初稿\n"
                f"围绕“{user_input}”，建议采用“目标人群痛点 → 商品卖点 → 使用场景 → 行动引导”的结构。\n\n"
                "**标题示例**：解决日常运营难题的实用好物，效率与体验兼顾\n"
                "**卖点表达**：聚焦真实商品属性、清晰说明适用场景，避免夸大功效或虚构优惠。\n\n"
                "请补充商品名称、核心属性、目标平台和目标人群，我可以继续生成可发布的版本。"
            )
        if any(keyword in user_input for keyword in ("活动", "方案", "双11", "双 11")):
            return (
                f"{prefix}## 活动方案骨架\n"
                f"活动主题：{user_input}\n\n"
                "1. 目标：明确 GMV、订单量或拉新目标；\n"
                "2. 货品：选择引流款、利润款和形象款；\n"
                "3. 节奏：预热、爆发、返场三个阶段；\n"
                "4. 复盘：按渠道、商品和转化漏斗评估。\n\n"
                "**建议下一步**：确认目标与预算、梳理货品池、制定内容排期。"
            )
        return (
            f"{prefix}我已收到运营需求：“{user_input}”。\n\n"
            "请补充时间范围、目标指标或商品/渠道信息。我会据此输出数据分析、运营建议或内容初稿。"
        )

    @staticmethod
    def _fragments(text: str) -> list[str]:
        # 短文本分片既能保证 SSE 交互及时，也能保持内容可读。
        size = 24
        return [text[index : index + size] for index in range(0, len(text), size)]
