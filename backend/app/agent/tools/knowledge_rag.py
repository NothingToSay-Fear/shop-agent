"""将运营资料知识库 RAG 封装为 Agent 工具。"""

from langchain_core.tools import BaseTool, tool

from app.agent.tools.tracker import AgentToolTracker
from app.agent.tools.registry import get_tool_specification
from app.config import Settings
from app.database import SessionLocal
from app.services.knowledge.retrieval import query_knowledge_for_question
from app.services.retrieval.query_expansion import PreparedRetrievalQueries


def build_knowledge_rag_tool(
    tracker: AgentToolTracker,
    settings: Settings,
    user_id: str | None = None,
    prepared_queries: PreparedRetrievalQueries | None = None,
) -> BaseTool:
    """创建受用户资料选择范围约束的知识库工具。"""

    @tool("query_knowledge_rag")
    async def query_knowledge_rag(question: str) -> str:
        """检索当前用户已选择参与问答的运营资料。适用于活动规则、玩法、商品资料、SOP、历史方案和复盘事实。必须传入用户原始问题。"""
        started_at = tracker.start_tool_call()
        input_summary = f"问题长度：{len(question.strip())} 个字符；检索范围：当前用户已选择资料"
        diagnostic_input = {
            "question_length": len(question.strip()),
            "has_prepared_queries": prepared_queries is not None,
        }
        specification = get_tool_specification("query_knowledge_rag")
        if not tracker.reserve_tool_call("query_knowledge_rag", specification.max_calls_per_run):
            tracker.record_tool_call(
                tool_name="query_knowledge_rag",
                input_summary=input_summary,
                result_summary="本轮知识库检索已执行，拒绝重复调用",
                status="skipped",
                started_at=started_at,
                error_code="tool_call_limit_reached",
            )
            return "本轮知识库检索已由系统执行，请基于已有受控结果回答，不要重复检索。"
        try:
            async with SessionLocal() as session:
                context = await query_knowledge_for_question(
                    session, question, user_id, settings=settings, prepared_queries=prepared_queries
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
                diagnostic_input=diagnostic_input,
            )
            raise

    return query_knowledge_rag
