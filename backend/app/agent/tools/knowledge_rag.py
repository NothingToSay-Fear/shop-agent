"""将运营资料知识库 RAG 封装为 Agent 工具。"""

from langchain_core.tools import BaseTool, tool

from app.agent.tools.tracker import AgentToolTracker
from app.config import Settings
from app.database import SessionLocal
from app.services.knowledge_rag import query_knowledge_for_question


def build_knowledge_rag_tool(tracker: AgentToolTracker, settings: Settings) -> BaseTool:
    """创建支持按资料分组检索的知识库工具。"""

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

    return query_knowledge_rag
