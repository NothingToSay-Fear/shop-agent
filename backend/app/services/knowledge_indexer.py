"""PostgreSQL 驱动的知识库后台索引任务，不依赖易丢失的进程内后台任务。"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, select, update

from app.config import Settings, get_settings
from app.database import SessionLocal
from app.models import KnowledgeChunk, KnowledgeDocument, KnowledgeIndexJob
from app.services.document_parser import parse_document
from app.services.local_embeddings import embed_texts

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


async def recover_interrupted_jobs() -> int:
    """Worker 重启后将遗留 running 任务放回队列，避免索引永久卡住。"""
    async with SessionLocal() as session:
        result = await session.execute(
            update(KnowledgeIndexJob)
            .where(KnowledgeIndexJob.status == "running")
            .values(status="queued", stage="queued", run_after=_now())
        )
        if result.rowcount:
            await session.execute(
                update(KnowledgeDocument)
                .where(KnowledgeDocument.status == "processing")
                .values(status="queued")
            )
        await session.commit()
        return result.rowcount or 0


async def claim_next_job(settings: Settings | None = None) -> str | None:
    """以行锁领取一项到期任务，多个 Worker 同时运行时不会重复处理。"""
    active_settings = settings or get_settings()
    async with SessionLocal() as session:
        job = await session.scalar(
            select(KnowledgeIndexJob)
            .where(KnowledgeIndexJob.status == "queued", KnowledgeIndexJob.run_after <= _now())
            .order_by(KnowledgeIndexJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        if job is None:
            return None
        document = await session.get(KnowledgeDocument, job.document_id)
        if document is None:
            await session.delete(job)
            await session.commit()
            return None
        job.status = "running"
        job.stage = "parsing"
        job.attempt_count += 1
        job.started_at = _now()
        job.error_message = None
        job.max_attempts = active_settings.knowledge_index_max_attempts
        document.status = "processing"
        document.error_message = None
        # 重试时先清空上一次失败任务留下的不可检索片段。
        await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id))
        await session.commit()
        return job.id


async def process_index_job(job_id: str, settings: Settings | None = None) -> None:
    """解析、分批向量化并持久化；任一批提交前都会检查资料是否已被删除。"""
    active_settings = settings or get_settings()
    try:
        document = await _load_active_document(job_id)
        if document is None:
            return
        raw_content = await asyncio.to_thread(Path(document.file_path).read_bytes)
        parsed = await asyncio.to_thread(parse_document, document.original_filename, raw_content)
        if not await _set_total_chunks(job_id, len(parsed.chunks)):
            return
        for offset in range(0, len(parsed.chunks), active_settings.knowledge_index_batch_size):
            batch = parsed.chunks[offset : offset + active_settings.knowledge_index_batch_size]
            embeddings = await embed_texts([chunk.content for chunk in batch], active_settings)
            if embeddings is None:
                raise RuntimeError("本地嵌入模型不可用")
            if not await _persist_chunk_batch(job_id, offset, batch, embeddings, active_settings):
                return
        await _mark_succeeded(job_id, parsed.content, len(parsed.chunks), active_settings)
    except Exception as error:
        logger.exception("knowledge_index_job_failed job_id=%s", job_id)
        await _mark_failed(job_id, error)


async def _load_active_document(job_id: str) -> KnowledgeDocument | None:
    async with SessionLocal() as session:
        job = await session.get(KnowledgeIndexJob, job_id)
        if job is None or job.status != "running":
            return None
        return await session.get(KnowledgeDocument, job.document_id)


async def _set_total_chunks(job_id: str, total_chunks: int) -> bool:
    async with SessionLocal() as session:
        job = await session.get(KnowledgeIndexJob, job_id)
        document = await session.get(KnowledgeDocument, job.document_id) if job else None
        if job is None or document is None or job.status != "running" or document.status != "processing":
            return False
        job.stage = "embedding"
        job.total_chunks = total_chunks
        job.processed_chunks = 0
        await session.commit()
        return True


async def _persist_chunk_batch(
    job_id: str, offset: int, chunks: list, embeddings: list[list[float]], settings: Settings
) -> bool:
    async with SessionLocal() as session:
        job = await session.get(KnowledgeIndexJob, job_id)
        document = await session.get(KnowledgeDocument, job.document_id) if job else None
        if job is None or document is None or job.status != "running" or document.status != "processing":
            return False
        session.add_all(
            KnowledgeChunk(
                document_id=document.id,
                chunk_index=offset + index,
                content=chunk.content,
                page_number=chunk.page_number,
                heading=chunk.heading,
                embedding=embedding,
                embedding_model=settings.local_embedding_model_id,
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        )
        job.processed_chunks = offset + len(chunks)
        await session.commit()
        return True


async def _mark_succeeded(job_id: str, content: str, chunk_count: int, settings: Settings) -> None:
    async with SessionLocal() as session:
        job = await session.get(KnowledgeIndexJob, job_id)
        document = await session.get(KnowledgeDocument, job.document_id) if job else None
        if job is None or document is None or job.status != "running" or document.status != "processing":
            return
        document.content = content
        document.chunk_count = chunk_count
        document.status = "ready"
        document.error_message = None
        job.status = "succeeded"
        job.stage = "completed"
        job.processed_chunks = chunk_count
        job.embedding_model = settings.local_embedding_model_id
        job.completed_at = _now()
        await session.commit()


async def _mark_failed(job_id: str, error: Exception) -> None:
    """失败时清理不可检索的半成品；暂时性错误最多按指数退避重试。"""
    async with SessionLocal() as session:
        job = await session.get(KnowledgeIndexJob, job_id)
        document = await session.get(KnowledgeDocument, job.document_id) if job else None
        if job is None:
            return
        if document is not None:
            await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id))
        message = str(error)[:500] or "索引任务失败"
        should_retry = not isinstance(error, ValueError) and job.attempt_count < job.max_attempts
        job.error_message = message
        if should_retry:
            job.status = "queued"
            job.stage = "retrying"
            job.run_after = _now() + timedelta(seconds=2 ** job.attempt_count)
            if document is not None:
                document.status = "queued"
                document.error_message = "索引暂时失败，正在重试"
        else:
            job.status = "failed"
            job.stage = "failed"
            job.completed_at = _now()
            if document is not None:
                document.status = "failed"
                document.error_message = message
        await session.commit()


async def run_worker() -> None:
    """持续领取任务；空闲时短暂休眠以避免轮询占满数据库连接。"""
    settings = get_settings()
    recovered = await recover_interrupted_jobs()
    logger.info("knowledge_index_worker_started recovered_jobs=%s", recovered)
    while True:
        job_id = await claim_next_job(settings)
        if job_id is None:
            await asyncio.sleep(settings.knowledge_index_poll_seconds)
            continue
        await process_index_job(job_id, settings)
