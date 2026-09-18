"""用户在会话中确认或忽略长期记忆候选的接口。"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import UTC, datetime

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Conversation, User, UserMemoryCandidate
from app.schemas.memory import UserMemoryCandidateRead
from app.services.authentication import get_current_user
from app.services.memory.user_memory import MemoryService

router = APIRouter(prefix="/api/memory-candidates", tags=["memory-candidates"])


@router.get("", response_model=list[UserMemoryCandidateRead])
async def list_memory_candidates(
    conversation_id: str | None = None,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> list[UserMemoryCandidate]:
    """返回当前用户在指定会话中尚未处理的候选。"""
    if conversation_id is not None:
        conversation = await session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.user_id == current_user.id
            )
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="会话不存在")
    statement = select(UserMemoryCandidate).where(
        UserMemoryCandidate.user_id == current_user.id,
        UserMemoryCandidate.status == "pending",
        or_(UserMemoryCandidate.expires_at.is_(None), UserMemoryCandidate.expires_at > datetime.now(UTC)),
    )
    if conversation_id is not None:
        statement = statement.where(UserMemoryCandidate.conversation_id == conversation_id)
    result = await session.scalars(statement.order_by(UserMemoryCandidate.created_at))
    return list(result)


@router.post("/{candidate_id}/accept", status_code=status.HTTP_204_NO_CONTENT)
async def accept_memory_candidate(
    candidate_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> None:
    """仅在用户明确点击后，才把候选写入可被 Agent 使用的记忆表。"""
    candidate = await _get_pending_candidate(candidate_id, current_user.id, session)
    try:
        await MemoryService.accept_candidate(session, candidate)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    await session.commit()


@router.post("/{candidate_id}/dismiss", status_code=status.HTTP_204_NO_CONTENT)
async def dismiss_memory_candidate(
    candidate_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> None:
    """忽略候选只更新候选状态，不会写入或影响任何已确认记忆。"""
    candidate = await _get_pending_candidate(candidate_id, current_user.id, session)
    MemoryService.dismiss_candidate(candidate)
    await session.commit()


async def _get_pending_candidate(
    candidate_id: str, user_id: str, session: AsyncSession
) -> UserMemoryCandidate:
    """用用户与 pending 状态共同约束候选，避免跨用户处理或重复确认。"""
    candidate = await session.scalar(
        select(UserMemoryCandidate).where(
            UserMemoryCandidate.id == candidate_id,
            UserMemoryCandidate.user_id == user_id,
            UserMemoryCandidate.status == "pending",
            or_(UserMemoryCandidate.expires_at.is_(None), UserMemoryCandidate.expires_at > datetime.now(UTC)),
        )
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="待确认记忆不存在或已处理")
    return candidate
