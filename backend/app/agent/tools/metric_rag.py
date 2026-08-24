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
        async with SessionLocal() as session:
            context = await query_metrics_for_question(session, question, settings=settings)
        tracker.metric_context = context
        if context is None:
            return "未检索到可用指标或对应数据，请说明数据不足，不要编造数值。"
        return context.text

    return query_metric_rag
