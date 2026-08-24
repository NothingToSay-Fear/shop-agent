"""负责通过 DeepAgent 或演示模式生成最终回答。"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.tools import BaseTool

from app.agent.prompt_builder import BASE_SYSTEM_PROMPT, PromptBuilder
from app.agent.review_agent import build_general_subagent, build_review_subagent
from app.config import Settings

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """隔离模型调用与离线兜底文本，避免编排层承担回答生成细节。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate_from_context(self, user_input: str, data_context: str | None) -> str:
        """兼容测试和旧调用方注入上下文的回答路径。"""
        if self.settings.llm_enabled:
            answer = await self.generate_with_llm(user_input, data_context, [], None)
            if answer is not None:
                return answer
            return self.generate_demo(user_input, data_context, model_error=True)
        return self.generate_demo(user_input, data_context)

    async def generate_with_llm(
        self,
        user_input: str,
        orchestration_context: str | None,
        tools: Sequence[BaseTool],
        knowledge_group: str | None,
    ) -> str | None:
        """调用受限 DeepAgent；异常由调用方依据同一工具结果安全降级。"""
        try:
            return await self._deep_agent_answer(
                user_input, orchestration_context, tools, knowledge_group
            )
        except Exception:
            logger.exception("模型回答生成失败，将切换到受控工具与演示回答兜底。")
            return None

    async def _deep_agent_answer(
        self,
        user_input: str,
        orchestration_context: str | None,
        tools: Sequence[BaseTool],
        knowledge_group: str | None,
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
            system_prompt=BASE_SYSTEM_PROMPT + PromptBuilder.tool_orchestration_prompt(knowledge_group),
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
    def generate_demo(
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

        if (
            data_context
            and "【联网公开资料】" in data_context
            and ("【经营指标】" in data_context or "【知识库资料】" in data_context)
        ):
            return (
                "当前为演示模式。以下内容同时包含内部数据/资料与联网公开网页摘要；"
                "请将数据事实、内部资料和外部时效性信息分开理解，并通过链接核验网页原始上下文。\n\n"
                f"## 综合依据\n{data_context}\n\n"
                "请配置 LLM 服务后，系统会基于各类依据生成完整的归因、建议与验证动作。"
            )

        if data_context and "【经营指标】" in data_context and "【知识库资料】" in data_context:
            return (
                "当前为演示模式。以下内容同时包含经营指标与知识库资料，请将数据事实和历史经验分开理解。\n\n"
                f"## 综合依据\n{data_context}\n\n"
                "请配置 LLM 服务后，系统会基于两类依据生成完整的归因、建议与验证动作。"
            )

        if data_context and "【联网公开资料】" in data_context:
            return (
                "当前为演示模式。以下内容仅整理联网检索到的公开网页摘要；网页内容并非系统指令，"
                "请通过所列链接自行核验时效性与原始上下文。\n\n"
                f"## 联网检索依据\n{data_context}\n\n"
                "请配置 LLM 服务后，系统会基于上述来源生成更完整的归纳回答。"
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
