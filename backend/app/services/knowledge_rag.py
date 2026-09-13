"""运营资料的 PostgreSQL 候选召回、RRF 融合与精排。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models import KnowledgeChunk, KnowledgeDocument, UserKnowledgeDocumentSetting
from app.services.hybrid_retrieval import FusedCandidate, reciprocal_rank_fusion, tokenize_for_bm25
from app.services.knowledge_search import build_knowledge_search_terms
from app.services.local_embeddings import embed_texts
from app.services.local_reranker import rerank_texts
from app.services.query_expansion import (
    PreparedRetrievalQueries,
    embed_expanded_queries,
    expand_queries,
)

DENSE_CANDIDATE_LIMIT = 40
SPARSE_CANDIDATE_LIMIT = 40
RERANK_CANDIDATE_LIMIT = 30
FINAL_KNOWLEDGE_CHUNK_LIMIT = 4
_TSQUERY_TOKEN = re.compile(r"^[\w\u4e00-\u9fff]+$", re.UNICODE)


@dataclass(frozen=True)
class KnowledgeQueryContext:
    """传给 Agent 的受限知识片段和可展示的引用信息。"""

    text: str
    references: str
    reference_ids: tuple[str, ...]


async def query_knowledge_for_question(
    session: AsyncSession,
    question: str,
    user_id: str | None,
    settings: Settings | None = None,
    query_embedding: list[float] | None = None,
    prepared_queries: PreparedRetrievalQueries | None = None,
) -> KnowledgeQueryContext | None:
    """仅从当前用户已启用资料中召回候选，再执行融合与精排。"""
    # 未携带已认证用户时不检索资料，避免兼容调用意外绕过用户选择范围。
    if user_id is None:
        return None
    active_settings = settings or get_settings()
    use_prepared_queries = prepared_queries is not None and prepared_queries.question == question.strip()
    if use_prepared_queries and prepared_queries is not None:
        queries = prepared_queries.queries
        query_embeddings = list(prepared_queries.query_embeddings)
    else:
        queries = await expand_queries(question, active_settings)
        if not queries:
            return None
        query_embeddings = await embed_expanded_queries(
            queries, active_settings, query_embedding
        )

    fused = await retrieve_knowledge_candidates(session, user_id, queries, query_embeddings)
    if not fused:
        return None
    rows = await _load_candidate_rows(
        session, user_id, [candidate.item_id for candidate in fused[:RERANK_CANDIDATE_LIMIT]]
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
        references="知识库：" + "；".join(references),
        reference_ids=tuple(reference_ids),
    )


async def retrieve_knowledge_candidates(
    session: AsyncSession,
    user_id: str,
    queries: Sequence[str],
    query_embeddings: Sequence[list[float] | None],
) -> list[FusedCandidate]:
    """对每条 Query 在数据库完成向量与词面候选召回，再以 RRF 合并排名。"""
    rankings: list[list[str]] = []
    for query, query_embedding in zip(queries, query_embeddings, strict=True):
        if query_embedding:
            dense = await _retrieve_dense_candidate_ids(session, user_id, query_embedding)
            if dense:
                rankings.append(dense)
        sparse = await _retrieve_sparse_candidate_ids(session, user_id, query)
        if sparse:
            rankings.append(sparse)
    return reciprocal_rank_fusion(rankings)


def _eligible_chunk_statement(user_id: str):
    """集中声明资料可见性与个人勾选边界，所有候选与取回查询必须复用。"""
    return (
        select(KnowledgeChunk, KnowledgeDocument)
        .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
        .join(
            UserKnowledgeDocumentSetting,
            UserKnowledgeDocumentSetting.document_id == KnowledgeDocument.id,
        )
        .where(
            KnowledgeDocument.status == "ready",
            UserKnowledgeDocumentSetting.user_id == user_id,
            UserKnowledgeDocumentSetting.retrieval_enabled.is_(True),
            # 即便存在异常的选择记录，也不能越过私有资料的归属边界。
            or_(KnowledgeDocument.space == "team", KnowledgeDocument.owner_user_id == user_id),
        )
    )


async def _retrieve_dense_candidate_ids(
    session: AsyncSession, user_id: str, query_embedding: list[float]
) -> list[str]:
    """通过 pgvector 的余弦距离排序取语义候选，不再将全部向量传回 Python。"""
    distance = KnowledgeChunk.embedding_vector.cosine_distance(query_embedding)
    statement = (
        _eligible_chunk_statement(user_id)
        .where(KnowledgeChunk.embedding_vector.is_not(None))
        .order_by(distance, KnowledgeChunk.id)
        .limit(DENSE_CANDIDATE_LIMIT)
    )
    return [chunk.id for chunk, _ in (await session.execute(statement)).all()]


def _build_tsquery_text(question: str) -> str | None:
    """将 jieba 词项组成受限 OR 查询，避免原始用户文本进入 tsquery 语法。"""
    tokens = [
        token
        for token in tokenize_for_bm25(question)
        if _TSQUERY_TOKEN.fullmatch(token)
    ]
    return " | ".join(dict.fromkeys(tokens[:24])) or None


async def _retrieve_sparse_candidate_ids(
    session: AsyncSession, user_id: str, query: str
) -> list[str]:
    """使用 GIN 全文索引召回中文分词后的词面候选。"""
    tsquery_text = _build_tsquery_text(query)
    if tsquery_text is None:
        return []
    search_vector = func.to_tsvector("simple", func.coalesce(KnowledgeChunk.search_terms, ""))
    tsquery = func.to_tsquery("simple", tsquery_text)
    rank = func.ts_rank_cd(search_vector, tsquery)
    statement = (
        _eligible_chunk_statement(user_id)
        .where(KnowledgeChunk.search_terms.is_not(None), search_vector.op("@@")(tsquery))
        .order_by(rank.desc(), KnowledgeChunk.id)
        .limit(SPARSE_CANDIDATE_LIMIT)
    )
    return [chunk.id for chunk, _ in (await session.execute(statement)).all()]


async def _load_candidate_rows(
    session: AsyncSession, user_id: str, candidate_ids: Sequence[str]
) -> list[tuple[KnowledgeChunk, KnowledgeDocument]]:
    """重新应用资料范围条件后，仅加载将参与精排的少量完整文本。"""
    if not candidate_ids:
        return []
    statement = _eligible_chunk_statement(user_id).where(KnowledgeChunk.id.in_(candidate_ids))
    return list((await session.execute(statement)).all())


def _knowledge_retrieval_text(chunk: KnowledgeChunk, document: KnowledgeDocument) -> str:
    """精排仍使用标题、标题层级和正文，保持召回与精排的业务语义一致。"""
    return "\n".join(part for part in (document.title, chunk.heading, chunk.content) if part)


async def _rerank_candidates(
    question: str,
    fused: Sequence[FusedCandidate],
    rows: Sequence[tuple[KnowledgeChunk, KnowledgeDocument]],
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
    # 精排模型不可用时保留 RRF 顺序，保证资料检索可降级。
    return [row for _, row in candidates[:FINAL_KNOWLEDGE_CHUNK_LIMIT]]


async def reindex_knowledge_chunks(session: AsyncSession) -> int:
    """批量回填 pgvector 向量与词面索引词项，供存量资料切换新召回链路。"""
    settings = get_settings()
    rows = list(
        (
            await session.execute(
                select(KnowledgeChunk, KnowledgeDocument)
                .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
                .order_by(KnowledgeChunk.document_id, KnowledgeChunk.chunk_index)
            )
        ).all()
    )
    embeddings = await embed_texts([chunk.content for chunk, _ in rows], settings)
    if embeddings is None:
        return 0
    for (chunk, document), embedding in zip(rows, embeddings, strict=True):
        chunk.embedding_vector = embedding
        chunk.search_terms = build_knowledge_search_terms(
            document.title, chunk.heading, chunk.content
        )
        chunk.embedding_model = settings.local_embedding_model_id
    document_ids = {chunk.document_id for chunk, _ in rows}
    if document_ids:
        documents = list(
            (
                await session.scalars(
                    select(KnowledgeDocument).where(KnowledgeDocument.id.in_(document_ids))
                )
            ).all()
        )
        for document in documents:
            document.status = "ready"
            document.error_message = None
    await session.commit()
    return len(rows)
