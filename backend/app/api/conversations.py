import asyncio
import json
import logging
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.operation_agent import OperationAgent
from app.agent.data_query_agent import DataQueryAgent
from app.agent.task_orchestrator import MainAgentOrchestrator
from app.agent.tools import AgentToolTracker
from app.agent.execution_plan import build_execution_plan_for_mode
from app.database import get_session
from app.models import (
    AgentRun,
    Conversation,
    ConversationContext,
    ConversationHistoryIndexJob,
    ConversationHistoryUnit,
    ConversationSummary,
    ConversationSummaryJob,
    ConversationTask,
    Message,
    ToolCall,
    User,
    UserMemoryCandidate,
)
from app.schemas.conversation import (
    AgentRunAuditRead,
    ConversationCreate,
    ConversationRead,
    MessageCreate,
    MessageRead,
)
from app.services.agent_audit import (
    complete_run,
    create_question_summary,
    fail_run,
    log_run_completed,
    log_run_failed,
    persist_tool_calls,
    update_run_route,
)
from app.services.conversation_history import (
    enqueue_history_unit_after_turn,
    retrieve_history_for_generation,
)
from app.services.conversation_summary import (
    retrieve_summary_for_generation,
    update_memory_state_after_turn,
)
from app.services.conversation_tasks import (
    TaskConstraintAudit,
    cancel_conversation_tasks,
    complete_conversation_task,
    prepare_conversation_task,
    task_constraint_audit,
)
from app.services.task_plans import (
    complete_task_plan,
    ensure_task_plan,
    fail_task_plan,
    start_task_plan,
)
from app.services.authentication import get_current_user
from app.services.user_memory import MemoryService

router = APIRouter(prefix="/api/conversations", tags=["conversations"])
logger = logging.getLogger("uvicorn.error")


@router.post("", response_model=ConversationRead, status_code=201)
async def create_conversation(
    payload: ConversationCreate,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> Conversation:
    """创建空会话，并在收到首条用户消息后更新标题。"""
    conversation = Conversation(title=payload.title, user_id=current_user.id)
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


@router.get("", response_model=list[ConversationRead])
async def list_conversations(
    session: AsyncSession = Depends(get_session), current_user: User = Depends(get_current_user)
) -> list[Conversation]:
    result = await session.scalars(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(result)


@router.get("/{conversation_id}/messages", response_model=list[MessageRead])
async def list_messages(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> list[Message]:
    """确认父会话存在后，按展示顺序返回消息。"""
    await _get_conversation(conversation_id, current_user.id, session)
    result = await session.scalars(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    )
    return list(result)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> None:
    """删除当前用户的无效会话及其从属消息、上下文和运行审计。"""
    await _get_conversation(conversation_id, current_user.id, session)
    running_run = await session.scalar(
        select(AgentRun.id)
        .where(AgentRun.conversation_id == conversation_id, AgentRun.status == "running")
        .limit(1)
    )
    if running_run is not None:
        raise HTTPException(status_code=409, detail="会话正在生成回答，完成后再删除")

    message_ids = select(Message.id).where(Message.conversation_id == conversation_id)
    run_ids = select(AgentRun.id).where(AgentRun.conversation_id == conversation_id)
    # 部分早期外键不具备级联删除，因此在受同一事务保护下按依赖逆序清理。
    await session.execute(
        delete(ToolCall).where(or_(ToolCall.message_id.in_(message_ids), ToolCall.run_id.in_(run_ids)))
    )
    # 待确认候选只属于原会话；已确认的长期记忆位于 user_memories，不能被会话删除影响。
    await session.execute(
        delete(UserMemoryCandidate).where(UserMemoryCandidate.conversation_id == conversation_id)
    )
    await session.execute(
        delete(ConversationSummaryJob).where(
            ConversationSummaryJob.conversation_id == conversation_id
        )
    )
    await session.execute(
        delete(ConversationSummary).where(ConversationSummary.conversation_id == conversation_id)
    )
    await session.execute(
        delete(ConversationHistoryIndexJob).where(
            ConversationHistoryIndexJob.unit_id.in_(
                select(ConversationHistoryUnit.id).where(
                    ConversationHistoryUnit.conversation_id == conversation_id
                )
            )
        )
    )
    await session.execute(
        delete(ConversationHistoryUnit).where(
            ConversationHistoryUnit.conversation_id == conversation_id
        )
    )
    await session.execute(delete(AgentRun).where(AgentRun.conversation_id == conversation_id))
    await session.execute(delete(Message).where(Message.conversation_id == conversation_id))
    await session.execute(
        delete(ConversationContext).where(ConversationContext.conversation_id == conversation_id)
    )
    await session.execute(
        delete(ConversationTask).where(ConversationTask.conversation_id == conversation_id)
    )
    await session.execute(
        delete(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == current_user.id
        )
    )
    await session.commit()


@router.delete("/{conversation_id}/context", status_code=204)
async def reset_conversation_context(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> None:
    """清除可继承条件；历史消息和运行审计保持不变。"""
    await _get_conversation(conversation_id, current_user.id, session)
    context = await session.get(ConversationContext, conversation_id)
    if context is not None:
        await session.delete(context)
    await cancel_conversation_tasks(session, conversation_id)
    await session.commit()


@router.post("/{conversation_id}/messages")
async def create_message(
    conversation_id: str,
    payload: MessageCreate,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """保存用户消息，并以 SSE 事件流返回对应的 Agent 回答。"""
    conversation = await _get_conversation(conversation_id, current_user.id, session)
    user_message = Message(
        conversation_id=conversation_id,
        sender_type="user",
        content=payload.content,
    )
    session.add(user_message)
    if conversation.title == "新会话":
        conversation.title = payload.content[:40]
    await session.commit()
    command_result = await MemoryService.handle_explicit_command(
        session, current_user.id, payload.content, conversation_id, user_message.id
    )
    if command_result is not None:
        return StreamingResponse(
            _memory_command_event_stream(session, conversation_id, command_result.message),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    task_turn = await prepare_conversation_task(
        session, conversation_id, user_message.id, payload.content
    )
    task_plan = await ensure_task_plan(session, task_turn.task)
    task_audit = task_constraint_audit(task_turn.task)
    if task_turn.requires_clarification:
        return StreamingResponse(
            _task_clarification_event_stream(
                session,
                conversation_id,
                user_message.id,
                task_audit,
                task_turn.clarification or "请补充必要条件。",
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    await start_task_plan(session, task_plan)
    data_query_agent = DataQueryAgent(task_turn.task.effective_constraints, payload.content)
    delegation_plan = MainAgentOrchestrator().plan(task_turn.task)
    memory_context = await MemoryService.retrieve_for_query(session, current_user.id, payload.content)
    await MemoryService.record_context_usage(session, current_user.id, memory_context)
    summary_context = await retrieve_summary_for_generation(session, conversation_id)
    history_context = await retrieve_history_for_generation(
        session, conversation_id, payload.content, summary_context
    )

    # 先提交运行起点，确保流式调用中断时仍有可排查的失败轨迹。
    run = AgentRun(
        conversation_id=conversation_id,
        user_message_id=user_message.id,
        question_summary=create_question_summary(payload.content),
        context_summary=task_audit.summary,
        context_actions=list(task_audit.actions),
        context_snapshot=task_audit.snapshot,
        memory_summary=memory_context.audit_summary,
        memory_ids=memory_context.ids,
        memory_selection=memory_context.audit_selection,
        conversation_summary_version=summary_context.version,
        conversation_summary_used=summary_context.used,
        conversation_history_ids=list(history_context.unit_ids),
        conversation_history_used=history_context.used,
    )
    session.add(run)
    await session.commit()

    agent = OperationAgent()
    started_at = perf_counter()

    async def event_stream():
        # 浏览器会立即渲染每个分片，并在收到 `done` 后重新加载持久化消息。
        full_answer = ""
        try:
            async for event in agent.stream_events(
                task_turn.effective_question,
                retrieval_mode=payload.mode,
                user_memory_context=memory_context,
                user_id=current_user.id,
                conversation_summary_context=summary_context,
                conversation_history_context=history_context,
                route_override=task_turn.route_override,
                data_query_agent=data_query_agent,
                delegation_plan=delegation_plan,
            ):
                if event.event_type == "status":
                    yield _event("status", {"content": event.content, "phase": event.phase or ""})
                    continue
                full_answer += event.content
                yield _event("chunk", {"content": event.content})
                await asyncio.sleep(0)
            agent_message = Message(
                conversation_id=conversation_id,
                sender_type="agent",
                content=full_answer,
                data_references=agent.data_references,
            )
            session.add(agent_message)
            await session.flush()
            if agent.tool_tracker.route is not None:
                update_run_route(run, agent.tool_tracker.route)
            await complete_conversation_task(task_turn.task, agent.tool_tracker.route)
            await complete_task_plan(session, task_plan, agent.tool_tracker.tool_calls)
            run.agent_message_id = agent_message.id
            complete_run(run, full_answer, agent.tool_tracker, started_at)
            persist_tool_calls(session, run, agent_message.id, agent.tool_tracker.tool_calls)
            await session.commit()
            log_run_completed(run, agent.tool_tracker)
            try:
                await update_memory_state_after_turn(
                    session,
                    conversation_id,
                    user_message.id,
                    payload.content,
                    agent_message.id,
                    full_answer,
                    run.id,
                )
            except Exception:
                # 短期记忆为辅助状态，更新失败不能影响本轮已完成的回答和审计记录。
                await session.rollback()
                logger.exception("conversation_memory_state_update_failed")
            try:
                await enqueue_history_unit_after_turn(
                    session,
                    conversation_id,
                    user_message.id,
                    payload.content,
                    agent_message.id,
                    full_answer,
                )
            except Exception:
                # 历史召回索引是连续性辅助能力，写入失败不能影响本轮已完成回答。
                await session.rollback()
                logger.exception("conversation_history_index_enqueue_failed")
            try:
                candidate = await MemoryService.create_candidate_if_eligible(
                    session,
                    current_user.id,
                    conversation_id,
                    user_message.id,
                    agent_message.id,
                    payload.content,
                )
            except Exception:
                # 候选确认不属于回答主链路；失败时不回滚已完成的 Agent 回答和审计。
                await session.rollback()
                logger.exception("memory_candidate_creation_failed")
                candidate = None
            if candidate is not None:
                yield _event(
                    "memory_candidate",
                    {
                        "candidate": {
                            "id": candidate.id,
                            "conversation_id": candidate.conversation_id,
                            "source_message_id": candidate.source_message_id,
                            "agent_message_id": candidate.agent_message_id,
                            "memory_type": candidate.memory_type,
                            "content": candidate.content,
                            "confidence": float(candidate.confidence),
                            "expires_at": candidate.expires_at.isoformat() if candidate.expires_at else None,
                            "created_at": candidate.created_at.isoformat(),
                        }
                    },
                )
            yield _event("done", {"message_id": agent_message.id, "run_id": run.id})
        except Exception:
            await session.rollback()
            await fail_task_plan(session, task_plan, "Agent 执行异常")
            if agent.tool_tracker.route is not None:
                update_run_route(run, agent.tool_tracker.route)
            fail_run(run, started_at)
            session.add(run)
            # 失败前已执行的工具同样需要审计；此时尚无 Agent 消息，关联本轮用户消息。
            persist_tool_calls(session, run, user_message.id, agent.tool_tracker.tool_calls)
            await session.commit()
            log_run_failed(run)
            yield _event("error", {"message": "生成回答失败，请稍后重试。"})

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@router.get("/{conversation_id}/messages/{message_id}/audit", response_model=AgentRunAuditRead)
async def get_message_audit(
    conversation_id: str,
    message_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> AgentRunAuditRead:
    """返回单条 Agent 回答的脱敏运行轨迹，供前端按需展开。"""
    await _get_conversation(conversation_id, current_user.id, session)
    run = await session.scalar(
        select(AgentRun).where(
            AgentRun.conversation_id == conversation_id,
            AgentRun.agent_message_id == message_id,
        )
    )
    if run is None:
        raise HTTPException(status_code=404, detail="未找到该回答的运行审计记录")
    execution_plan = build_execution_plan_for_mode(run.route_mode)
    calls = list(
        await session.scalars(
            select(ToolCall).where(ToolCall.run_id == run.id).order_by(ToolCall.created_at)
        )
    )
    return AgentRunAuditRead(
        id=run.id,
        question_summary=run.question_summary,
        route_mode=run.route_mode,
        route_confidence=float(run.route_confidence) if run.route_confidence is not None else None,
        route_fallback=run.route_fallback,
        context_summary=run.context_summary,
        context_actions=run.context_actions,
        context_snapshot=run.context_snapshot,
        memory_summary=run.memory_summary,
        memory_ids=run.memory_ids,
        memory_selection=run.memory_selection,
        conversation_summary_version=run.conversation_summary_version,
        conversation_summary_used=run.conversation_summary_used,
        conversation_history_ids=run.conversation_history_ids,
        conversation_history_used=run.conversation_history_used,
        status=run.status,
        answer_summary=run.answer_summary,
        reference_ids=run.reference_ids,
        execution_plan=list(execution_plan.required_tools) if execution_plan else [],
        total_duration_ms=run.total_duration_ms,
        error_code=run.error_code,
        created_at=run.created_at,
        completed_at=run.completed_at,
        tool_calls=[
            {
                "tool_name": call.tool_name,
                "input_summary": call.input_summary,
                "result_summary": call.result_summary,
                "reference_ids": call.reference_ids,
                "status": call.status,
                "duration_ms": call.duration_ms,
                "error_code": call.error_code,
                "created_at": call.created_at,
            }
            for call in calls
        ],
    )


async def _get_conversation(
    conversation_id: str, user_id: str, session: AsyncSession
) -> Conversation:
    """只返回当前用户拥有的会话，避免以 ID 枚举其他用户的数据。"""
    conversation = await session.scalar(
        select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conversation


async def _memory_command_event_stream(
    session: AsyncSession, conversation_id: str, response: str
):
    """将显式记忆命令作为普通 Agent 消息返回，但不创建无意义的 AgentRun。"""
    agent_message = Message(
        conversation_id=conversation_id,
        sender_type="agent",
        content=response,
        data_references="用户长期记忆",
        status="memory_command",
    )
    session.add(agent_message)
    await session.commit()
    yield _event("chunk", {"content": response})
    yield _event("done", {"message_id": agent_message.id})


async def _task_clarification_event_stream(
    session: AsyncSession,
    conversation_id: str,
    user_message_id: str,
    task_audit: TaskConstraintAudit,
    response: str,
):
    """把受控的任务补充请求作为普通回答持久化，不提前执行不完整的 RAG 查询。"""
    # 追问也是一条可追溯的 Agent 输出，但没有执行任何检索工具。
    run = AgentRun(
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        question_summary="等待补充条件的会话任务",
        route_mode="task_clarification",
        context_summary=task_audit.summary,
        context_actions=list(task_audit.actions),
        context_snapshot=task_audit.snapshot,
    )
    session.add(run)
    agent_message = Message(
        conversation_id=conversation_id,
        sender_type="agent",
        content=response,
        data_references="会话任务状态：等待用户补充条件",
        status="task_clarification",
    )
    session.add(agent_message)
    await session.flush()
    run.agent_message_id = agent_message.id
    complete_run(run, response, AgentToolTracker(), perf_counter())
    await session.commit()
    yield _event("status", {"content": "正在等待补充当前任务所需条件…", "phase": "task"})
    yield _event("chunk", {"content": response})
    yield _event("done", {"message_id": agent_message.id})


def _event(event: str, payload: dict[str, object]) -> str:
    """编码单条服务端推送事件，避免 Agent 层感知传输协议细节。"""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
