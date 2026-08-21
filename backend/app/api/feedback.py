from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Feedback, Message

router = APIRouter(prefix="/api/messages", tags=["feedback"])


class FeedbackCreate(BaseModel):
    """单条有帮助/无帮助回答反馈的请求体。"""
    feedback_type: str = Field(pattern="^(up|down)$")
    reason: str | None = Field(default=None, max_length=1000)


@router.post("/{message_id}/feedback", status_code=201)
async def create_feedback(
    message_id: str, payload: FeedbackCreate, session: AsyncSession = Depends(get_session)
) -> dict[str, str]:
    """仅在关联回答存在时记录反馈。"""
    if await session.get(Message, message_id) is None:
        raise HTTPException(status_code=404, detail="消息不存在")
    feedback = Feedback(message_id=message_id, **payload.model_dump())
    session.add(feedback)
    await session.commit()
    return {"id": feedback.id, "message": "反馈已记录"}
