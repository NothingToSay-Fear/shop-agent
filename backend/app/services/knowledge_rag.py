"""运营资料的本地向量检索与引用上下文构建。"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models import KnowledgeChunk, KnowledgeDocument
from app.services.local_embeddings import embed_texts
from app.services.metric_rag import cosine_similarity


@dataclass(frozen=True)
class KnowledgeQueryContext:
    """传给 Agent 的受限知识片段和可展示的引用信息。"""

    text: str
    references: str
    reference_ids: tuple[str, ...]


async def query_knowledge_for_question(
    session: AsyncSession,
    question: str,
    group_name: str | None = None,
    settings: Settings | None = None,
    query_embedding: list[float] | None = None,
) -> KnowledgeQueryContext | None:
    """仅检索已完成向量化的片段，避免以未处理资料作为问答依据。"""
    active_settings = settings or get_settings()
    active_query_embedding = query_embedding
    if active_query_embedding is None:
        embeddings = await embed_texts([question], active_settings)
        active_query_embedding = embeddings[0] if embeddings else None
    if active_query_embedding is None:
        return None

    statement = (
        select(KnowledgeChunk, KnowledgeDocument)
        .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
        .where(KnowledgeDocument.status == "ready", KnowledgeChunk.embedding.is_not(None))
    )
    if group_name:
        statement = statement.where(KnowledgeDocument.group_name == group_name)
    rows = (await session.execute(statement)).all()
    scored = sorted(
        (
            (cosine_similarity(active_query_embedding, chunk.embedding or []), chunk, document)
            for chunk, document in rows
            if chunk.embedding and len(chunk.embedding) == len(active_query_embedding)
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    selected = [item for item in scored[:4] if item[0] >= 0.35]
    if not selected:
        return None

    text_parts = ["以下内容来自知识库。仅可依据这些资料回答；资料未提及的内容请明确说明。"]
    references: list[str] = []
    reference_ids: list[str] = []
    for _, chunk, document in selected:
        location = f"第 {chunk.page_number} 页" if chunk.page_number else f"片段 {chunk.chunk_index + 1}"
        text_parts.append(f"【{document.title}｜{location}】\n{chunk.content}")
        reference = f"{document.title}（{location}）"
        if reference not in references:
            references.append(reference)
        reference_id = f"knowledge_chunk:{chunk.id}"
        if reference_id not in reference_ids:
            reference_ids.append(reference_id)
    return KnowledgeQueryContext(
        text="\n\n".join(text_parts),
        references="知识库：" + "、".join(references),
        reference_ids=tuple(reference_ids),
    )


async def reindex_knowledge_chunks(session: AsyncSession) -> int:
    """在本地嵌入模型下载或切换后，批量重建全部资料片段向量。"""
    settings = get_settings()
    chunks = list((await session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.document_id, KnowledgeChunk.chunk_index))).all())
    embeddings = await embed_texts([chunk.content for chunk in chunks], settings)
    if embeddings is None:
        return 0
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk.embedding = embedding
        chunk.embedding_model = settings.local_embedding_model_id
    document_ids = {chunk.document_id for chunk in chunks}
    if document_ids:
        documents = list((await session.scalars(select(KnowledgeDocument).where(KnowledgeDocument.id.in_(document_ids)))).all())
        for document in documents:
            document.status = "ready"
            document.error_message = None
    await session.commit()
    return len(chunks)
