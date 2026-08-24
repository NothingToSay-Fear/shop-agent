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
        started_at = tracker.start_tool_call()
        group_summary = group_name.strip() if group_name else "全部分组"
        input_summary = f"问题长度：{len(question.strip())} 个字符；资料分组：{group_summary}"
        try:
            async with SessionLocal() as session:
                context = await query_knowledge_for_question(
                    session, question, group_name, settings=settings
                )
            tracker.knowledge_context = context
            tracker.knowledge_miss = context is None
            if context is None:
                tracker.record_tool_call(
                    tool_name="query_knowledge_rag",
                    input_summary=input_summary,
                    result_summary="未命中可靠知识库资料",
                    status="empty",
                    started_at=started_at,
                )
                return "知识库中未检索到可靠资料，请明确说明资料不足，不要补充未知事实。"
            tracker.record_tool_call(
                tool_name="query_knowledge_rag",
                input_summary=input_summary,
                result_summary=f"命中 {len(context.reference_ids)} 个知识库片段",
                reference_ids=context.reference_ids,
                status="success",
                started_at=started_at,
            )
            return context.text
        except Exception:
            tracker.record_tool_call(
                tool_name="query_knowledge_rag",
                input_summary=input_summary,
                result_summary="知识库检索执行失败",
                status="failed",
                started_at=started_at,
                error_code="knowledge_rag_failed",
            )
            raise

    return query_knowledge_rag
