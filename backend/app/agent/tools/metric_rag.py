"""将受控经营指标 RAG 封装为 Agent 工具。"""

from datetime import date

from langchain_core.tools import BaseTool, tool

from app.agent.tools.tracker import AgentToolTracker
from app.agent.tools.registry import get_tool_specification
from app.config import Settings
from app.database import SessionLocal
from app.services.metric_rag import (
    MetricQueryConstraints,
    MetricQueryPlanError,
    query_metrics_for_question,
)


def build_metric_rag_tool(tracker: AgentToolTracker, settings: Settings) -> BaseTool:
    """创建仅能执行已审核只读 SQL 模板的指标查询工具。"""

    @tool("query_metric_rag")
    async def query_metric_rag(
        question: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> str:
        """查询经营指标。日期条件只能由系统确认后作为结构化参数传入；工具只执行经审核的只读 SQL 模板。"""
        started_at = tracker.start_tool_call()
        input_summary = f"问题长度：{len(question.strip())} 个字符"
        specification = get_tool_specification("query_metric_rag")
        if not tracker.reserve_tool_call("query_metric_rag", specification.max_calls_per_run):
            tracker.record_tool_call(
                tool_name="query_metric_rag",
                input_summary=input_summary,
                result_summary="本轮指标查询已执行，拒绝重复调用",
                status="skipped",
                started_at=started_at,
                error_code="tool_call_limit_reached",
            )
            return "本轮指标查询已由系统执行，请基于已有受控结果回答，不要重复检索。"
        try:
            async with SessionLocal() as session:
                context = await query_metrics_for_question(
                    session,
                    question,
                    settings=settings,
                    constraints=MetricQueryConstraints(start_date=start_date, end_date=end_date),
                )
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
        except MetricQueryPlanError as error:
            tracker.record_tool_call(
                tool_name="query_metric_rag",
                input_summary=input_summary,
                result_summary=str(error),
                status="empty",
                started_at=started_at,
                error_code="metric_query_plan_invalid",
            )
            return str(error)
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
