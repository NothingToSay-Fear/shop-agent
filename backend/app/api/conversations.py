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
from app.services.knowledge_rag import query_knowledge_for_question
from app.services.intent_router import route_question

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
    # 综合模式先以本地语义路由选择数据源；低置信度问题才保守地同时检索两类资料。
    metric_context = None
    knowledge_context = None
    effective_mode = payload.mode
    query_embedding = None
    if payload.mode == "hybrid":
        route = await route_question(payload.content)
        effective_mode = route.mode
        query_embedding = route.query_embedding
    if effective_mode in ("hybrid", "metrics"):
        metric_context = await query_metrics_for_question(
            session, payload.content, query_embedding=query_embedding
        )
    if effective_mode in ("hybrid", "knowledge"):
        knowledge_context = await query_knowledge_for_question(
            session, payload.content, payload.knowledge_group, query_embedding=query_embedding
        )
    context_parts: list[str] = []
    if metric_context:
        context_parts.append(f"【经营指标】\n{metric_context.text}")
    if knowledge_context:
        context_parts.append(f"【知识库资料】\n{knowledge_context.text}")
    if effective_mode == "knowledge" and not knowledge_context:
        context_parts.append("【知识库资料】\n知识库中未检索到足以回答该问题的资料，请明确说明资料不足。")
    data_context = "\n\n".join(context_parts) or None

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
                data_references=_references(metric_context, knowledge_context, effective_mode),
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


def _references(metric_context, knowledge_context, mode: str) -> str:
    """将指标与资料来源并列保存，供历史会话复核。"""
    references: list[str] = []
    if metric_context:
        references.append(f"内置模拟经营数据：{', '.join(metric_context.metric_codes)}")
    if knowledge_context:
        references.append(knowledge_context.references)
    if references:
        return "；".join(references)
    if mode == "knowledge":
        return "知识库未检索到相关资料"
    return "演示模式：尚未接入真实数据源或相关资料"
