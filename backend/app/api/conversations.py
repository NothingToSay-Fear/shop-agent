import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.operation_agent import OperationAgent
from app.database import get_session
from app.models import Conversation, Message
from app.schemas.conversation import (
    ConversationCreate,
    ConversationRead,
    MessageCreate,
    MessageRead,
)
from app.services.metric_rag import query_metrics_for_question

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.post("", response_model=ConversationRead, status_code=201)
async def create_conversation(
    payload: ConversationCreate, session: AsyncSession = Depends(get_session)
) -> Conversation:
    """创建空会话，并在收到首条用户消息后更新标题。"""
    conversation = Conversation(title=payload.title)
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


@router.get("", response_model=list[ConversationRead])
async def list_conversations(session: AsyncSession = Depends(get_session)) -> list[Conversation]:
    result = await session.scalars(select(Conversation).order_by(Conversation.updated_at.desc()))
    return list(result)


@router.get("/{conversation_id}/messages", response_model=list[MessageRead])
async def list_messages(
    conversation_id: str, session: AsyncSession = Depends(get_session)
) -> list[Message]:
    """确认父会话存在后，按展示顺序返回消息。"""
    await _get_conversation(conversation_id, session)
    result = await session.scalars(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    )
    return list(result)


@router.post("/{conversation_id}/messages")
async def create_message(
    conversation_id: str,
    payload: MessageCreate,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """保存用户消息，并以 SSE 事件流返回对应的 Agent 回答。"""
    conversation = await _get_conversation(conversation_id, session)
    user_message = Message(
        conversation_id=conversation_id,
        sender_type="user",
        content=payload.content,
    )
    session.add(user_message)
    if conversation.title == "新会话":
        conversation.title = payload.content[:40]
    await session.commit()

    agent = OperationAgent()
    # RAG 先从指标知识表中匹配用户问题，再只执行该指标及依赖所需的受控 SQL。
    metric_context = await query_metrics_for_question(session, payload.content)
    data_context = metric_context.text if metric_context else None

    async def event_stream():
        # 浏览器会立即渲染每个分片，并在收到 `done` 后重新加载持久化消息。
        full_answer = ""
        try:
            async for fragment in agent.stream(payload.content, data_context):
                full_answer += fragment
                yield _event("chunk", {"content": fragment})
                await asyncio.sleep(0)
            agent_message = Message(
                conversation_id=conversation_id,
                sender_type="agent",
                content=full_answer,
                data_references=(
                    f"内置模拟经营数据：{', '.join(metric_context.metric_codes)}"
                    if metric_context
                    else "演示模式：尚未接入真实数据源"
                ),
            )
            session.add(agent_message)
            await session.commit()
            yield _event("done", {"message_id": agent_message.id})
        except Exception:
            await session.rollback()
            yield _event("error", {"message": "生成回答失败，请稍后重试。"})

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


async def _get_conversation(conversation_id: str, session: AsyncSession) -> Conversation:
    """将未知会话 ID 转换为统一的 HTTP 404 响应。"""
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conversation


def _event(event: str, payload: dict[str, str]) -> str:
    """编码单条服务端推送事件，避免 Agent 层感知传输协议细节。"""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
