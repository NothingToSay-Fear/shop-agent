"""将受控 RAG 服务封装为 DeepAgent 可调用的异步工具。"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool, tool

from app.config import Settings
from app.database import SessionLocal
from app.services.knowledge_rag import KnowledgeQueryContext, query_knowledge_for_question
from app.services.metric_rag import MetricQueryContext, query_metrics_for_question


@dataclass
class RagToolTracker:
    """收集一次 Agent 运行中实际调用的 RAG 依据，供会话消息持久化。"""

    metric_context: MetricQueryContext | None = None
    knowledge_context: KnowledgeQueryContext | None = None
    knowledge_miss: bool = False

    @property
    def data_context(self) -> str | None:
        """将工具检索结果组装为演示模式与最终回答共用的受限上下文。"""
        parts: list[str] = []
        if self.metric_context:
            parts.append(f"【经营指标】\n{self.metric_context.text}")
        if self.knowledge_context:
            parts.append(f"【知识库资料】\n{self.knowledge_context.text}")
        if self.knowledge_miss and not self.knowledge_context:
            parts.append("【知识库资料】\n知识库中未检索到足以回答该问题的资料，请明确说明资料不足。")
        return "\n\n".join(parts) or None

    @property
    def references(self) -> str:
        """返回实际检索到的来源，避免把未调用的能力写入会话记录。"""
        references: list[str] = []
        if self.metric_context:
            references.append(f"内置模拟经营数据：{', '.join(self.metric_context.metric_codes)}")
        if self.knowledge_context:
            references.append(self.knowledge_context.references)
        if references:
            return "；".join(references)
        if self.knowledge_miss:
            return "知识库未检索到相关资料"
        return "演示模式：尚未检索到相关数据或资料"


def build_rag_tools(tracker: RagToolTracker, settings: Settings) -> list[BaseTool]:
    """创建仅能访问受控服务的工具，不向 Agent 暴露数据库连接或任意 SQL。"""

    @tool("query_metric_rag")
    async def query_metric_rag(question: str) -> str:
        """查询经营指标。适用于 GMV、订单、访客、转化率、客单价、退款、趋势及指定日期范围问题。必须传入用户原始问题，工具只执行经审核的只读 SQL 模板。"""
        async with SessionLocal() as session:
            context = await query_metrics_for_question(session, question, settings=settings)
        tracker.metric_context = context
        if context is None:
            return "未检索到可用指标或对应数据，请说明数据不足，不要编造数值。"
        return context.text

    @tool("query_knowledge_rag")
    async def query_knowledge_rag(question: str, group_name: str | None = None) -> str:
        """检索运营资料知识库。适用于活动规则、玩法、商品资料、SOP、历史方案和复盘事实。可选 group_name 用于限定用户选择的资料分组。必须传入用户原始问题。"""
        async with SessionLocal() as session:
            context = await query_knowledge_for_question(
                session, question, group_name, settings=settings
            )
        tracker.knowledge_context = context
        tracker.knowledge_miss = context is None
        if context is None:
            return "知识库中未检索到可靠资料，请明确说明资料不足，不要补充未知事实。"
        return context.text

    return [query_metric_rag, query_knowledge_rag]
