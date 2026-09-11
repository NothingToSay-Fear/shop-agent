"""运营资料的本地向量检索与引用上下文构建。"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models import KnowledgeChunk, KnowledgeDocument
from app.services.hybrid_retrieval import FusedCandidate, hybrid_retrieve
from app.services.local_embeddings import embed_texts
from app.services.local_reranker import rerank_texts
from app.services.metric_rag import cosine_similarity
from app.services.query_expansion import embed_expanded_queries, expand_queries

RERANK_CANDIDATE_LIMIT = 30
FINAL_KNOWLEDGE_CHUNK_LIMIT = 4


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
    """从已就绪资料中进行多 Query 混合召回，并只返回最可靠的少量片段。"""
    active_settings = settings or get_settings()
    statement = (
        select(KnowledgeChunk, KnowledgeDocument)
        .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
        .where(KnowledgeDocument.status == "ready", KnowledgeChunk.embedding.is_not(None))
    )
    if group_name:
        statement = statement.where(KnowledgeDocument.group_name == group_name)
    rows = (await session.execute(statement)).all()
    if not rows:
        return None
    queries = await expand_queries(question, active_settings)
    if not queries:
        return None
    query_embeddings = await embed_expanded_queries(
        queries, active_settings, query_embedding
    )
    fused = hybrid_retrieve(
        rows,
        queries,
        query_embeddings,
        get_id=lambda row: row[0].id,
        get_text=lambda row: _knowledge_retrieval_text(row[0], row[1]),
        get_embedding=lambda row: row[0].embedding,
        dense_score=lambda row, _query, vector: cosine_similarity(vector, row[0].embedding or []),
    )
    selected = await _rerank_candidates(question, fused, rows, active_settings)
    if not selected:
        return None

    text_parts = ["以下内容来自知识库。仅可依据这些资料回答；资料未提及的内容请明确说明。"]
    references: list[str] = []
    reference_ids: list[str] = []
    for chunk, document in selected:
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


def _knowledge_retrieval_text(chunk: KnowledgeChunk, document: KnowledgeDocument) -> str:
    """标题、标题层级和正文共同参与 BM25，提升活动名与规则词的精确召回。"""
    return "\n".join(part for part in (document.title, chunk.heading, chunk.content) if part)


async def _rerank_candidates(
    question: str,
    fused: list[FusedCandidate],
    rows: list[tuple[KnowledgeChunk, KnowledgeDocument]],
    settings: Settings,
) -> list[tuple[KnowledgeChunk, KnowledgeDocument]]:
    """在 RRF 去重后的候选上精排，并过滤低于校准阈值的片段。"""
    row_by_id = {chunk.id: (chunk, document) for chunk, document in rows}
    candidates = [
        (candidate, row_by_id[candidate.item_id])
        for candidate in fused[:RERANK_CANDIDATE_LIMIT]
        if candidate.item_id in row_by_id
    ]
    if not candidates:
        return []
    scores = await rerank_texts(
        question,
        [_knowledge_retrieval_text(chunk, document) for _, (chunk, document) in candidates],
        settings,
    )
    if scores is not None:
        candidates = [
            item
            for score, item in sorted(
                zip(scores, candidates, strict=True),
                key=lambda pair: (-pair[0], -pair[1][0].rrf_score, pair[1][0].item_id),
            )
            if score >= settings.knowledge_reranker_min_score
        ]
    # 精排模型不可用时 scores 为 None，保留既有 RRF 顺序作为可用性降级。
    return [row for _, row in candidates[:FINAL_KNOWLEDGE_CHUNK_LIMIT]]


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
