"""知识库文件的上传、浏览、下载与分组接口。"""

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_session
from app.models import KnowledgeChunk, KnowledgeDocument
from app.schemas.knowledge import KnowledgeDocumentContent, KnowledgeDocumentRead, KnowledgeGroupRead
from app.services.document_parser import SUPPORTED_FILE_TYPES, parse_document
from app.services.local_embeddings import embed_texts

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])
logger = logging.getLogger(__name__)


@router.get("/groups", response_model=list[KnowledgeGroupRead])
async def list_groups(session: AsyncSession = Depends(get_session)) -> list[KnowledgeGroupRead]:
    """返回已存在分组，供上传和检索范围选择。"""
    result = await session.execute(
        select(KnowledgeDocument.group_name, func.count(KnowledgeDocument.id))
        .group_by(KnowledgeDocument.group_name)
        .order_by(KnowledgeDocument.group_name)
    )
    return [KnowledgeGroupRead(name=name, document_count=count) for name, count in result.all()]


@router.get("/documents", response_model=list[KnowledgeDocumentRead])
async def list_documents(
    group_name: str | None = None, session: AsyncSession = Depends(get_session)
) -> list[KnowledgeDocument]:
    """按可选分组列出资料，默认返回全部。"""
    statement = select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc())
    if group_name:
        statement = statement.where(KnowledgeDocument.group_name == group_name)
    return list((await session.scalars(statement)).all())


@router.post("/documents", response_model=KnowledgeDocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    group_name: str = Form(...),
    session: AsyncSession = Depends(get_session),
) -> KnowledgeDocument:
    """保存原文件，解析切块，并在模型可用时立即完成向量化。"""
    settings = get_settings()
    clean_group = group_name.strip()
    if not clean_group or len(clean_group) > 100:
        raise HTTPException(status_code=422, detail="分组名称长度应为 1 至 100 个字符")
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
    try:
        parsed = await asyncio.to_thread(parse_document, original_filename, raw_content)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=422, detail="文件解析失败，请确认文件未损坏且不含加密") from error

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
        title=Path(original_filename).stem[:200] or "未命名资料",
        original_filename=original_filename[:255],
        file_type=suffix.lstrip("."),
        file_path=str(storage_path),
        group_name=clean_group,
        content=parsed.content,
        status="pending_embedding",
        chunk_count=len(parsed.chunks),
    )
    embeddings = await embed_texts([chunk.content for chunk in parsed.chunks], settings)
    chunks = [
        KnowledgeChunk(
            document_id=document_id,
            chunk_index=index,
            content=chunk.content,
            page_number=chunk.page_number,
            heading=chunk.heading,
            embedding=embeddings[index] if embeddings else None,
            embedding_model=settings.local_embedding_model_id if embeddings else None,
        )
        for index, chunk in enumerate(parsed.chunks)
    ]
    if embeddings:
        document.status = "ready"
    session.add(document)
    try:
        # 当前模型未声明 ORM relationship；先 flush 父记录，确保片段外键指向已存在的文档。
        await session.flush()
        session.add_all(chunks)
        await session.commit()
        await session.refresh(document)
    except Exception as error:
        await session.rollback()
        # 原文件不能在数据库写入失败后长期成为不可见的孤立文件。
        await asyncio.to_thread(storage_path.unlink, missing_ok=True)
        raise HTTPException(status_code=500, detail="知识库文件入库失败，请稍后重试") from error
    return document


@router.get("/documents/{document_id}", response_model=KnowledgeDocumentContent)
async def get_document(document_id: str, session: AsyncSession = Depends(get_session)) -> KnowledgeDocument:
    """返回提取后的文本，用于浏览器内的统一预览。"""
    return await _get_document(document_id, session)


@router.get("/documents/{document_id}/download")
async def download_document(document_id: str, session: AsyncSession = Depends(get_session)) -> FileResponse:
    """下载未经改写的原始上传文件。"""
    document = await _get_document(document_id, session)
    path = Path(document.file_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="原始文件不存在")
    return FileResponse(path, filename=document.original_filename, media_type="application/octet-stream")


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, session: AsyncSession = Depends(get_session)) -> None:
    """删除资料、其级联片段向量和原始文件，避免过期资料影响检索。"""
    document = await _get_document(document_id, session)
    path = Path(document.file_path)
    await session.delete(document)
    await session.commit()
    try:
        await asyncio.to_thread(path.unlink, missing_ok=True)
    except OSError as error:
        # 数据库记录已删除，原件残留不再可访问；记录日志供运维后续清理。
        logger.warning("知识库原始文件删除失败：%s", path, exc_info=error)


async def _get_document(document_id: str, session: AsyncSession) -> KnowledgeDocument:
    document = await session.get(KnowledgeDocument, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="知识库文件不存在")
    return document
