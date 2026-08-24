"""将受控经营指标 RAG 封装为 Agent 工具。"""

from langchain_core.tools import BaseTool, tool

from app.agent.tools.tracker import AgentToolTracker
from app.config import Settings
from app.database import SessionLocal
from app.services.metric_rag import query_metrics_for_question


def build_metric_rag_tool(tracker: AgentToolTracker, settings: Settings) -> BaseTool:
    """创建仅能执行已审核只读 SQL 模板的指标查询工具。"""

    @tool("query_metric_rag")
    async def query_metric_rag(question: str) -> str:
        """查询经营指标。适用于 GMV、订单、访客、转化率、客单价、退款、趋势及指定日期范围问题。必须传入用户原始问题，工具只执行经审核的只读 SQL 模板。"""
        started_at = tracker.start_tool_call()
        input_summary = f"问题长度：{len(question.strip())} 个字符"
        try:
            async with SessionLocal() as session:
                context = await query_metrics_for_question(session, question, settings=settings)
            tracker.metric_context = context
            if context is None:
                tracker.record_tool_call(
                    tool_name="query_metric_rag",
                    input_summary=input_summary,
                    result_summary="未命中可用指标或对应数据",
                    status="empty",
                    started_at=started_at,
                )
                return "未检索到可用指标或对应数据，请说明数据不足，不要编造数值。"
            reference_ids = tuple(f"metric:{code}" for code in context.metric_codes)
            tracker.record_tool_call(
                tool_name="query_metric_rag",
                input_summary=input_summary,
                result_summary=f"命中 {len(reference_ids)} 个经营指标",
                reference_ids=reference_ids,
                status="success",
                started_at=started_at,
            )
            return context.text
        except Exception:
            tracker.record_tool_call(
                tool_name="query_metric_rag",
                input_summary=input_summary,
                result_summary="指标查询执行失败",
                status="failed",
                started_at=started_at,
                error_code="metric_rag_failed",
            )
            raise

    return query_metric_rag
