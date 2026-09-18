"""知识库文件的上传、浏览与下载接口。"""

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_session
from app.models import KnowledgeDocument, KnowledgeIndexJob, User, UserKnowledgeDocumentSetting
from app.schemas.knowledge import (
    KnowledgeDocumentContent,
    KnowledgeDocumentRead,
    KnowledgeDocumentRetrievalUpdate,
)
from app.services.authentication import get_current_user
from app.services.knowledge.document_parser import SUPPORTED_FILE_TYPES

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])
logger = logging.getLogger(__name__)


@router.get("/documents", response_model=list[KnowledgeDocumentRead])
async def list_documents(
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> list[KnowledgeDocumentRead]:
    """列出当前用户可见的资料。"""
    statement = (
        select(KnowledgeDocument)
        .where(_visible_document_condition(current_user))
        .order_by(KnowledgeDocument.created_at.desc())
    )
    documents = list((await session.scalars(statement)).all())
    jobs = await _latest_jobs(session, [document.id for document in documents])
    enabled_ids = await _retrieval_enabled_document_ids(
        session, current_user.id, [document.id for document in documents]
    )
    return [
        _serialize_document(document, jobs.get(document.id), document.id in enabled_ids)
        for document in documents
    ]


@router.post("/documents", response_model=KnowledgeDocumentRead, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    space: str = Form("private"),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> KnowledgeDocumentRead:
    """快速保存原文件并创建持久化任务；耗时索引由独立 Worker 完成。"""
    settings = get_settings()
    clean_space = space.strip().lower()
    if clean_space not in {"private", "team"}:
        raise HTTPException(status_code=422, detail="资料空间只能是 private 或 team")
    if clean_space == "team" and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="仅管理员可以上传团队资料")
    original_filename = file.filename or "untitled"
    suffix = Path(original_filename).suffix.lower()
    if suffix not in SUPPORTED_FILE_TYPES:
        raise HTTPException(status_code=415, detail="仅支持 PDF、DOCX、Markdown 和 TXT 文件")
    raw_content = await file.read()
    maximum_size = settings.knowledge_max_upload_size_mb * 1024 * 1024
    if not raw_content:
        raise HTTPException(status_code=422, detail="不能上传空文件")
    if len(raw_content) > maximum_size:
        raise HTTPException(status_code=413, detail=f"文件不能超过 {settings.knowledge_max_upload_size_mb}MB")
    document_id = str(uuid.uuid4())
    storage_dir = Path(settings.knowledge_upload_dir)
    storage_path = storage_dir / f"{document_id}{suffix}"
    try:
        await asyncio.to_thread(storage_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(storage_path.write_bytes, raw_content)
    except OSError as error:
        raise HTTPException(status_code=500, detail="无法保存上传文件") from error

    document = KnowledgeDocument(
        id=document_id,
        owner_user_id=current_user.id,
        space=clean_space,
        title=Path(original_filename).stem[:200] or "未命名资料",
        original_filename=original_filename[:255],
        file_type=suffix.lstrip("."),
        file_path=str(storage_path),
        # 解析正文由后台任务补齐；空字符串保证上传接口不会等待耗时解析。
        content="",
        status="queued",
        chunk_count=0,
    )
    job = KnowledgeIndexJob(
        document_id=document_id,
        status="queued",
        stage="queued",
        max_attempts=settings.knowledge_index_max_attempts,
    )
    session.add(document)
    session.add(job)
    # 私有资料属于上传者，默认加入其问答检索；团队资料必须由每位用户自行勾选。
    if clean_space == "private":
        session.add(
            UserKnowledgeDocumentSetting(
                user_id=current_user.id, document_id=document_id, retrieval_enabled=True
            )
        )
    try:
        # 文档与任务在同一事务提交，避免出现“已保存却永远不会被索引”的孤立文件记录。
        await session.flush()
        await session.commit()
        await session.refresh(document)
    except Exception as error:
        await session.rollback()
        # 原文件不能在数据库写入失败后长期成为不可见的孤立文件。
        await asyncio.to_thread(storage_path.unlink, missing_ok=True)
        raise HTTPException(status_code=500, detail="知识库文件入库失败，请稍后重试") from error
    return _serialize_document(document, job, retrieval_enabled=clean_space == "private")


@router.get("/documents/{document_id}", response_model=KnowledgeDocumentContent)
async def get_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> KnowledgeDocumentContent:
    """返回提取后的文本，用于浏览器内的统一预览。"""
    document = await _get_visible_document(document_id, current_user, session)
    job = (await _latest_jobs(session, [document.id])).get(document.id)
    enabled_ids = await _retrieval_enabled_document_ids(session, current_user.id, [document.id])
    return KnowledgeDocumentContent(
        **_serialize_document(document, job, document.id in enabled_ids).model_dump(), content=document.content
    )


@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    """下载未经改写的原始上传文件。"""
    document = await _get_visible_document(document_id, current_user, session)
    path = Path(document.file_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="原始文件不存在")
    return FileResponse(path, filename=document.original_filename, media_type="application/octet-stream")


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> None:
    """删除资料、其级联片段向量和原始文件，避免过期资料影响检索。"""
    document = await _get_visible_document(document_id, current_user, session)
    if document.space == "team" and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="仅管理员可以删除团队资料")
    path = Path(document.file_path)
    await session.delete(document)
    await session.commit()
    try:
        await asyncio.to_thread(path.unlink, missing_ok=True)
    except OSError as error:
        # 数据库记录已删除，原件残留不再可访问；记录日志供运维后续清理。
        logger.warning("知识库原始文件删除失败：%s", path, exc_info=error)


@router.put("/documents/{document_id}/retrieval", response_model=KnowledgeDocumentRead)
async def update_document_retrieval(
    document_id: str,
    payload: KnowledgeDocumentRetrievalUpdate,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> KnowledgeDocumentRead:
    """由当前用户独立控制一份可见资料是否参与自己的知识库问答。"""
    document = await _get_visible_document(document_id, current_user, session)
    setting = await session.get(UserKnowledgeDocumentSetting, (current_user.id, document.id))
    if setting is None:
        setting = UserKnowledgeDocumentSetting(
            user_id=current_user.id,
            document_id=document.id,
            retrieval_enabled=payload.retrieval_enabled,
        )
        session.add(setting)
    else:
        setting.retrieval_enabled = payload.retrieval_enabled
    await session.commit()
    job = (await _latest_jobs(session, [document.id])).get(document.id)
    return _serialize_document(document, job, payload.retrieval_enabled)


def _visible_document_condition(user: User):
    """团队资料对所有登录用户可见；私有资料只对上传者可见。"""
    return or_(KnowledgeDocument.space == "team", KnowledgeDocument.owner_user_id == user.id)


async def _get_visible_document(
    document_id: str, current_user: User, session: AsyncSession
) -> KnowledgeDocument:
    document = await session.scalar(
        select(KnowledgeDocument).where(
            KnowledgeDocument.id == document_id, _visible_document_condition(current_user)
        )
    )
    if document is None:
        raise HTTPException(status_code=404, detail="知识库文件不存在")
    return document


async def _retrieval_enabled_document_ids(
    session: AsyncSession, user_id: str, document_ids: list[str]
) -> set[str]:
    """只读取当前用户自己的勾选记录，避免资料选择状态在用户之间串扰。"""
    if not document_ids:
        return set()
    result = await session.scalars(
        select(UserKnowledgeDocumentSetting.document_id).where(
            UserKnowledgeDocumentSetting.user_id == user_id,
            UserKnowledgeDocumentSetting.document_id.in_(document_ids),
            UserKnowledgeDocumentSetting.retrieval_enabled.is_(True),
        )
    )
    return set(result)


async def _latest_jobs(
    session: AsyncSession, document_ids: list[str]
) -> dict[str, KnowledgeIndexJob]:
    """每份资料只向前端暴露最新任务，避免历史重试记录干扰进度展示。"""
    if not document_ids:
        return {}
    jobs = list(
        (
            await session.scalars(
                select(KnowledgeIndexJob)
                .where(KnowledgeIndexJob.document_id.in_(document_ids))
                .order_by(KnowledgeIndexJob.created_at.desc())
            )
        ).all()
    )
    latest: dict[str, KnowledgeIndexJob] = {}
    for job in jobs:
        latest.setdefault(job.document_id, job)
    return latest


def _serialize_document(
    document: KnowledgeDocument, job: KnowledgeIndexJob | None, retrieval_enabled: bool = False
) -> KnowledgeDocumentRead:
    """将任务进度与资料元数据组合成前端可轮询的单一响应。"""
    values = KnowledgeDocumentRead.model_validate(document).model_dump()
    values.update(
        index_status=job.status if job else None,
        index_stage=job.stage if job else None,
        processed_chunks=job.processed_chunks if job else 0,
        total_chunks=job.total_chunks if job else 0,
        index_error_message=job.error_message if job else None,
        retrieval_enabled=retrieval_enabled,
    )
    return KnowledgeDocumentRead(**values)
