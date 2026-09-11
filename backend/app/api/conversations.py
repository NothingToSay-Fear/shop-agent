import asyncio
import json
import logging
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.operation_agent import OperationAgent
from app.agent.execution_plan import build_execution_plan_for_mode
from app.database import get_session
from app.models import AgentRun, Conversation, ConversationContext, Message, ToolCall, User
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
from app.services.conversation_context import build_and_persist_context
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
        session, current_user.id, payload.content
    )
    if command_result is not None:
        return StreamingResponse(
            _memory_command_event_stream(session, conversation_id, command_result.message),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    context_result = await build_and_persist_context(
        session,
        conversation_id,
        user_message.id,
        payload.content,
        payload.knowledge_group,
    )
    memory_context = await MemoryService.retrieve_for_query(session, current_user.id, payload.content)
    await MemoryService.record_context_usage(session, current_user.id, memory_context)

    # 先提交运行起点，确保流式调用中断时仍有可排查的失败轨迹。
    run = AgentRun(
        conversation_id=conversation_id,
        user_message_id=user_message.id,
        question_summary=create_question_summary(payload.content),
        context_summary=context_result.audit_summary,
        context_actions=list(context_result.audit_actions),
        context_snapshot=context_result.snapshot.as_audit_snapshot(),
        memory_summary=memory_context.audit_summary,
        memory_ids=memory_context.ids,
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
                payload.content,
                knowledge_group=payload.knowledge_group,
                retrieval_mode=payload.mode,
                conversation_context=context_result.snapshot,
                user_memory_context=memory_context,
                user_id=current_user.id,
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
            run.agent_message_id = agent_message.id
            complete_run(run, full_answer, agent.tool_tracker, started_at)
            persist_tool_calls(session, run, agent_message.id, agent.tool_tracker.tool_calls)
            await session.commit()
            log_run_completed(run, agent.tool_tracker)
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
                            "created_at": candidate.created_at.isoformat(),
                        }
                    },
                )
            yield _event("done", {"message_id": agent_message.id, "run_id": run.id})
        except Exception:
            await session.rollback()
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


def _event(event: str, payload: dict[str, object]) -> str:
    """编码单条服务端推送事件，避免 Agent 层感知传输协议细节。"""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
