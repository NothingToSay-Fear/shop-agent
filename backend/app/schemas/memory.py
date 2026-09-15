"""会话内长期记忆候选的 API 数据结构。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


MemoryType = Literal[
    "analysis_preference",
    "answer_preference",
    "focus_topic",
    "work_profile",
    "focus_direction",
]
class UserMemoryCandidateRead(BaseModel):
    """仅返回当前用户仍可确认的候选，不暴露已生效记忆的管理入口。"""

    id: str
    memory_type: MemoryType
    content: str
    conversation_id: str
    source_message_id: str
    agent_message_id: str
    confidence: float
    expires_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}
