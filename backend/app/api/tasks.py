from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Task
from app.schemas.task import TaskCreate, TaskRead, TaskUpdate

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get("", response_model=list[TaskRead])
async def list_tasks(session: AsyncSession = Depends(get_session)) -> list[Task]:
    result = await session.scalars(select(Task).order_by(Task.created_at.desc()))
    return list(result)


@router.post("", response_model=TaskRead, status_code=201)
async def create_task(payload: TaskCreate, session: AsyncSession = Depends(get_session)) -> Task:
    """手动创建任务，或从给出建议的消息创建任务。"""
    task = Task(**payload.model_dump())
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


@router.patch("/{task_id}", response_model=TaskRead)
async def update_task(
    task_id: str, payload: TaskUpdate, session: AsyncSession = Depends(get_session)
) -> Task:
    """仅更新请求中明确提供的字段，未提供的任务字段保持不变。"""
    task = await session.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(task, field, value)
    await session.commit()
    await session.refresh(task)
    return task
