from time import perf_counter

import pytest

from app.agent.tools.tracker import AgentToolTracker
from app.agent.tools.web_search import build_web_search_tool
from app.config import Settings
from app.services.agent_audit import (
    complete_run,
    create_question_summary,
    log_run_completed,
    log_run_failed,
    log_run_started,
    update_run_route,
)
from app.services.intent_router import RetrievalRoute
from app.services.conversations.task_plans import _safe_error_code


def test_question_summary_does_not_copy_original_question() -> None:
    """运行审计只能保存长度摘要，不能把用户原文再次写入审计表。"""
    question = "请分析 618 活动期间的 GMV，并给出投放建议"

    summary = create_question_summary(question)

    assert summary == f"用户运营问题（{len(question)} 个字符）"
    assert question not in summary


def test_tracker_aggregates_only_actual_reference_ids() -> None:
    """运行级引用只来自已记录的工具调用，并且会去重。"""
    tracker = AgentToolTracker()
    started_at = tracker.start_tool_call()
    tracker.record_tool_call(
        tool_name="query_metric_rag",
        input_summary="问题长度：10 个字符",
        result_summary="命中 2 个经营指标",
        reference_ids=("metric:paid_gmv", "metric:visitor_count"),
        status="success",
        started_at=started_at,
    )
    tracker.record_tool_call(
        tool_name="query_knowledge_rag",
        input_summary="问题长度：10 个字符；检索范围：当前用户已选择资料",
        result_summary="命中 1 个知识库片段",
        reference_ids=("metric:paid_gmv", "knowledge_chunk:chunk-1"),
        status="success",
        started_at=started_at,
    )

    assert tracker.reference_ids == [
        "metric:paid_gmv",
        "metric:visitor_count",
        "knowledge_chunk:chunk-1",
    ]


def test_completed_run_uses_route_and_summary_without_answer_copy() -> None:
    """完成状态应保存路由与长度摘要，而非 Agent 回答正文。"""
    from app.models import AgentRun

    run = AgentRun(question_summary="用户运营问题（10 个字符）")
    route = RetrievalRoute("web", None, 0.9, False)
    tracker = AgentToolTracker()
    update_run_route(run, route)

    answer = "这是只应保存在 messages 表中的完整回答。"
    complete_run(run, answer, tracker, perf_counter())

    assert run.route_mode == "web"
    assert run.route_confidence == 0.9
    assert run.status == "completed"
    assert answer not in (run.answer_summary or "")
    assert "0 个引用" in (run.answer_summary or "")


def test_runtime_logs_are_searchable_by_conversation_message_and_run_ids(caplog: pytest.LogCaptureFixture) -> None:
    """排障日志只能包含定位 ID、状态和摘要，不能包含用户或回答原文。"""
    from app.models import AgentRun

    run = AgentRun(
        id="run-001",
        conversation_id="conversation-001",
        user_message_id="user-message-001",
        agent_message_id="agent-message-001",
        question_summary="用户运营问题（12 个字符）",
        status="failed",
        total_duration_ms=320,
        error_code="agent_run_failed",
    )
    tracker = AgentToolTracker(run_id=run.id)
    with caplog.at_level("INFO", logger="uvicorn.error"):
        log_run_started(run, task_id="task-001", plan_id="plan-001", route_mode="metrics")
        tracker.record_tool_call(
            tool_name="query_metric_rag",
            input_summary="问题长度：12 个字符",
            result_summary="指标查询执行失败",
            status="failed",
            started_at=perf_counter(),
            error_code="metric_rag_failed",
            diagnostic_input={
                "metric_codes": ["paid_gmv"],
                "query_periods": [{"start_date": "2026-06-01", "end_date": "2026-06-18"}],
            },
        )
        log_run_failed(run, exception_type="ConnectionError")
        log_run_completed(run, tracker)

    output = "\n".join(caplog.messages)
    assert "conversation_id=conversation-001" in output
    assert "user_message_id=user-message-001" in output
    assert "agent_message_id=agent-message-001" in output
    assert "run_id=run-001" in output
    assert "tool_call_failed run_id=run-001" in output
    assert '"metric_codes": ["paid_gmv"]' in output
    assert '"start_date": "2026-06-01"' in output
    assert "exception_type=ConnectionError" in output


def test_plan_log_rejects_non_code_error_text() -> None:
    """计划日志不得把未来传入的异常正文作为可检索字段输出。"""
    assert _safe_error_code("database_unavailable") == "database_unavailable"
    assert _safe_error_code("连接失败：包含用户输入") == "step_failed"


@pytest.mark.asyncio
async def test_disabled_web_search_is_recorded_as_skipped() -> None:
    """未配置联网密钥时也要留下可解释的工具审计，而不是静默跳过。"""
    tracker = AgentToolTracker()
    tool = build_web_search_tool(tracker, Settings(web_search_api_key=None))

    result = await tool.ainvoke({"question": "北京今天天气如何？"})

    assert "尚未配置" in result
    assert tracker.tool_calls[0].status == "skipped"
    assert tracker.tool_calls[0].error_code == "web_search_disabled"
