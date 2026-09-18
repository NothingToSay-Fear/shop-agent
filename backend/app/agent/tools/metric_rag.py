"""将受控经营指标 RAG 封装为 Agent 工具。"""

from datetime import date

from langchain_core.tools import BaseTool, tool

from app.agent.tools.tracker import AgentToolTracker
from app.agent.tools.registry import get_tool_specification
from app.config import Settings
from app.database import SessionLocal
from app.services.metric_rag import (
    MetricQueryConstraints,
    MetricQueryPlan,
    MetricQueryPlanError,
    MetricQueryUnit,
    query_metrics_for_codes,
    query_metrics_for_question,
)
from app.services.query_expansion import PreparedRetrievalQueries


def build_metric_rag_tool(
    tracker: AgentToolTracker,
    settings: Settings,
    prepared_queries: PreparedRetrievalQueries | None = None,
) -> BaseTool:
    """创建仅能执行已审核只读 SQL 模板的指标查询工具。"""

    @tool("query_metric_rag")
    async def query_metric_rag(
        question: str,
        start_date: date | None = None,
        end_date: date | None = None,
        metric_codes: list[str] | None = None,
        query_periods: list[dict[str, str]] | None = None,
        dimensions: list[str] | None = None,
        capability_notes: list[str] | None = None,
    ) -> str:
        """查询经营指标。日期条件只能由系统确认后作为结构化参数传入；工具只执行经审核的只读 SQL 模板。"""
        started_at = tracker.start_tool_call()
        input_summary = f"问题长度：{len(question.strip())} 个字符"
        if metric_codes:
            input_summary += f"；数据查询计划指标={','.join(metric_codes)}"
        if dimensions:
            input_summary += f"；下钻维度={','.join(dimensions)}"
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
                constraints = MetricQueryConstraints(start_date=start_date, end_date=end_date)
                if metric_codes:
                    context = await query_metrics_for_codes(
                        session,
                        tuple(metric_codes),
                        question,
                        constraints=constraints,
                        query_plan=_query_plan_from_periods(query_periods),
                        dimensions=tuple(dimensions or ()),
                        capability_notes=tuple(capability_notes or ()),
                    )
                else:
                    context = await query_metrics_for_question(
                        session,
                        question,
                        settings=settings,
                        constraints=constraints,
                        prepared_queries=prepared_queries,
                    )
            if context is not None and tracker.metric_context is not None:
                previous = tracker.metric_context
                tracker.metric_context = type(context)(
                    text=f"{previous.text}\n\n【补充指标查询】\n{context.text}",
                    metric_codes=tuple(dict.fromkeys((*previous.metric_codes, *context.metric_codes))),
                    query_units=tuple(dict.fromkeys((*previous.query_units, *context.query_units))),
                )
            else:
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


def _query_plan_from_periods(
    query_periods: list[dict[str, str]] | None,
) -> MetricQueryPlan | None:
    """只接受任务状态已确认的 ISO 日期，拒绝把原始文本拼进查询。"""
    if not query_periods:
        return None
    units: list[MetricQueryUnit] = []
    for item in query_periods:
        label = item.get("label")
        start_date = item.get("start_date")
        end_date = item.get("end_date")
        if not all(isinstance(value, str) and value for value in (label, start_date, end_date)):
            raise MetricQueryPlanError("数据查询计划包含无效时间范围。")
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
        except ValueError as error:
            raise MetricQueryPlanError("数据查询计划包含无效日期。") from error
        if start > end:
            raise MetricQueryPlanError("数据查询计划的开始日期晚于结束日期。")
        units.append(MetricQueryUnit(label, start, end))
    if len(units) > 4:
        raise MetricQueryPlanError("一次最多支持 4 个明确时间范围，请拆分后再查询。")
    return MetricQueryPlan(tuple(units))
