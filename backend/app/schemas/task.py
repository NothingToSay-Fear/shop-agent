from datetime import datetime

from pydantic import BaseModel, Field


class TaskCreate(BaseModel):
    """将 Agent 建议转为待办事项时接受的任务字段。"""
    title: str = Field(min_length=1, max_length=200)
    source_message_id: str | None = None
    assignee: str | None = Field(default=None, max_length=100)
    due_date: datetime | None = None
    priority: str = Field(default="medium", pattern="^(low|medium|high)$")
    acceptance_metric: str | None = Field(default=None, max_length=500)


class TaskUpdate(BaseModel):
    """任务部分更新请求；字段按设计均为可选。"""
    title: str | None = Field(default=None, min_length=1, max_length=200)
    assignee: str | None = Field(default=None, max_length=100)
    due_date: datetime | None = None
    priority: str | None = Field(default=None, pattern="^(low|medium|high)$")
    status: str | None = Field(default=None, pattern="^(todo|in_progress|completed)$")
    acceptance_metric: str | None = Field(default=None, max_length=500)


class TaskRead(BaseModel):
    """返回给运营工作台的任务数据结构。"""
    id: str
    source_message_id: str | None
    title: str
    assignee: str | None
    due_date: datetime | None
    priority: str
    status: str
    acceptance_metric: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
