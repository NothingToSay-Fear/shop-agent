"""Agent 运行审计的摘要、持久化和过期清理。"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from time import perf_counter

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.tracker import AgentToolTracker, ToolCallAudit
from app.models import AgentRun, ToolCall
from app.services.intent_router import RetrievalRoute

# 复用 Uvicorn 已配置的标准输出处理器，确保 Docker 日志可检索运行轨迹。
logger = logging.getLogger("uvicorn.error")

AUDIT_RETENTION_DAYS = 15


def create_question_summary(question: str) -> str:
    """只记录问题长度，避免将原始提问写入运行审计表。"""
    return f"用户运营问题（{len(question.strip())} 个字符）"


def create_answer_summary(answer: str, reference_count: int) -> str:
    """只记录回答长度和来源数量，避免复制 Agent 原文。"""
    return f"Agent 已生成回答（{len(answer.strip())} 个字符，{reference_count} 个引用）"


def update_run_route(run: AgentRun, route: RetrievalRoute) -> None:
    """写入已实际采用的路由结果，不保存用户问题文本。"""
    run.route_mode = route.mode
    run.route_confidence = route.confidence
    run.route_fallback = route.fallback_to_hybrid
    run.question_summary = f"{route.mode} 路由的用户运营问题"


def complete_run(
    run: AgentRun,
    answer: str,
    tracker: AgentToolTracker,
    started_at: float,
) -> None:
    """以最小摘要结束一次成功运行。"""
    run.status = "completed"
    run.answer_summary = create_answer_summary(answer, len(tracker.reference_ids))
    run.reference_ids = tracker.reference_ids
    run.total_duration_ms = _duration_ms(started_at)
    run.completed_at = datetime.now(UTC)


def fail_run(run: AgentRun, started_at: float, error_code: str = "agent_run_failed") -> None:
    """标记失败但不写入异常原文，避免异常中意外包含敏感输入。"""
    run.status = "failed"
    run.error_code = error_code
    run.total_duration_ms = _duration_ms(started_at)
    run.completed_at = datetime.now(UTC)


def persist_tool_calls(
    session: AsyncSession, run: AgentRun, message_id: str, tool_calls: list[ToolCallAudit]
) -> None:
    """将内存轨迹转换为数据库明细，工具原始输入和输出不入库。"""
    for item in tool_calls:
        session.add(
            ToolCall(
                message_id=message_id,
                run_id=run.id,
                tool_name=item.tool_name,
                input_summary=item.input_summary,
                result_summary=item.result_summary,
                reference_ids=list(item.reference_ids),
                status=item.status,
                duration_ms=item.duration_ms,
                error_code=item.error_code,
            )
        )


async def cleanup_expired_agent_audits(session: AsyncSession) -> int:
    """删除超过 15 天的运行记录，关联工具调用由数据库外键级联删除。"""
    expires_at = datetime.now(UTC) - timedelta(days=AUDIT_RETENTION_DAYS)
    result = await session.execute(delete(AgentRun).where(AgentRun.created_at < expires_at))
    await session.commit()
    deleted_count = int(result.rowcount or 0)
    logger.info("agent_audit_cleanup retention_days=%s deleted_runs=%s", AUDIT_RETENTION_DAYS, deleted_count)
    return deleted_count


def log_run_completed(run: AgentRun, tracker: AgentToolTracker) -> None:
    """输出可由 Docker 日志检索的 key=value 运行摘要。"""
    logger.info(
        "agent_run_completed run_id=%s conversation_id=%s user_message_id=%s agent_message_id=%s route=%s status=%s duration_ms=%s tool_calls=%s references=%s",
        run.id,
        run.conversation_id,
        run.user_message_id,
        run.agent_message_id or "none",
        run.route_mode,
        run.status,
        run.total_duration_ms,
        len(tracker.tool_calls),
        len(run.reference_ids or []),
    )


def log_run_started(
    run: AgentRun,
    *,
    task_id: str | None = None,
    plan_id: str | None = None,
    route_mode: str | None = None,
) -> None:
    """记录可由前端已有会话或消息 ID 检索到的运行入口。"""
    logger.info(
        "agent_run_started run_id=%s conversation_id=%s user_message_id=%s task_id=%s plan_id=%s route=%s",
        run.id,
        run.conversation_id,
        run.user_message_id,
        task_id or "none",
        plan_id or "none",
        route_mode or run.route_mode or "unknown",
    )


def log_run_failed(
    run: AgentRun, *, stage: str = "workflow", exception_type: str | None = None
) -> None:
    """输出不含原始输入和异常正文的失败摘要。"""
    logger.warning(
        "agent_run_failed run_id=%s conversation_id=%s user_message_id=%s stage=%s status=%s duration_ms=%s error_code=%s exception_type=%s",
        run.id,
        run.conversation_id,
        run.user_message_id,
        stage,
        run.status,
        run.total_duration_ms,
        run.error_code,
        exception_type or "none",
    )


def _duration_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))
