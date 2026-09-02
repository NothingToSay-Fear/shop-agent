from datetime import datetime

from typing import Literal

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    """创建新会话时使用的请求体。"""
    title: str = Field(default="新会话", min_length=1, max_length=200)


class ConversationRead(BaseModel):
    """返回给 Web 客户端的会话数据结构。"""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    """发送给 Agent 的已校验用户输入。"""
    content: str = Field(min_length=1, max_length=4000)
    mode: Literal["hybrid", "metrics", "knowledge", "web"] = "hybrid"
    knowledge_group: str | None = Field(default=None, max_length=100)


class MessageRead(BaseModel):
    """会话历史中返回的已存储用户或 Agent 消息。"""
    id: str
    conversation_id: str
    sender_type: str
    content: str
    data_references: str | None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ToolCallAuditRead(BaseModel):
    """前端可查看的单次工具调用脱敏审计摘要。"""

    tool_name: str
    input_summary: str | None
    result_summary: str | None
    reference_ids: list[str]
    status: str
    duration_ms: int | None
    error_code: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentRunAuditRead(BaseModel):
    """一条回答对应的最小运行轨迹，不含原始问题和回答正文。"""

    id: str
    question_summary: str
    route_mode: str | None
    route_confidence: float | None
    route_fallback: bool
    context_summary: str | None
    context_actions: list[str]
    context_snapshot: dict[str, object]
    memory_summary: str | None
    memory_ids: list[str]
    status: str
    answer_summary: str | None
    reference_ids: list[str]
    execution_plan: list[str]
    total_duration_ms: int | None
    error_code: str | None
    created_at: datetime
    completed_at: datetime | None
    tool_calls: list[ToolCallAuditRead]
