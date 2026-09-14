"""运营资料的 PostgreSQL 候选召回、RRF 融合与精排。"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models import KnowledgeChunk, KnowledgeDocument, UserKnowledgeDocumentSetting
from app.services.document_parser import apply_semantic_boundaries, parse_document
from app.services.hybrid_retrieval import FusedCandidate, reciprocal_rank_fusion, tokenize_for_bm25
from app.services.knowledge_search import (
    build_knowledge_retrieval_text,
    build_knowledge_search_terms,
)
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
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgeQueryContext:
    """传给 Agent 的受限知识片段和可展示的引用信息。"""

    text: str
    references: str
    reference_ids: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeRetrievalTrace:
    """供离线评测读取的检索阶段轨迹，不写入线上用户审计记录。"""

    queries: tuple[str, ...]
    dense_rankings: tuple[tuple[str, ...], ...]
    sparse_rankings: tuple[tuple[str, ...], ...]
    fused_ranking: tuple[str, ...]
    final_ranking: tuple[str, ...]


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
        location = _format_chunk_location(chunk)
        heading_context = f"｜{chunk.heading_path}" if chunk.heading_path else ""
        text_parts.append(f"【{document.title}{heading_context}｜{location}】\n{chunk.content}")
        reference = f"{document.title}{heading_context}（{location}）"
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


async def trace_knowledge_retrieval(
    session: AsyncSession,
    user_id: str,
    question: str,
    settings: Settings | None = None,
) -> KnowledgeRetrievalTrace:
    """执行与线上一致的候选召回和精排，并暴露各阶段排名用于离线测评。"""
    active_settings = settings or get_settings()
    queries = await expand_queries(question, active_settings)
    embeddings = await embed_expanded_queries(queries, active_settings, None)
    dense_rankings: list[tuple[str, ...]] = []
    sparse_rankings: list[tuple[str, ...]] = []
    fusion_inputs: list[list[str]] = []
    for query, embedding in zip(queries, embeddings, strict=True):
        if embedding:
            dense = await _retrieve_dense_candidate_ids(session, user_id, embedding)
            dense_rankings.append(tuple(dense))
            if dense:
                fusion_inputs.append(dense)
        else:
            dense_rankings.append(())
        sparse = await _retrieve_sparse_candidate_ids(session, user_id, query)
        sparse_rankings.append(tuple(sparse))
        if sparse:
            fusion_inputs.append(sparse)
    fused = reciprocal_rank_fusion(fusion_inputs)
    rows = await _load_candidate_rows(
        session, user_id, [candidate.item_id for candidate in fused[:RERANK_CANDIDATE_LIMIT]]
    )
    selected = await _rerank_candidates(question, fused, rows, active_settings)
    return KnowledgeRetrievalTrace(
        queries=queries,
        dense_rankings=tuple(dense_rankings),
        sparse_rankings=tuple(sparse_rankings),
        fused_ranking=tuple(candidate.item_id for candidate in fused),
        final_ranking=tuple(chunk.id for chunk, _ in selected),
    )


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
    return build_knowledge_retrieval_text(
        document.title,
        getattr(chunk, "heading_path", None) or chunk.heading,
        getattr(chunk, "content_type", "paragraph"),
        chunk.content,
    )


def _format_chunk_location(chunk: KnowledgeChunk) -> str:
    """优先显示结构化页码范围，非 PDF 资料则回退到稳定片段序号。"""
    if chunk.page_start is not None:
        if chunk.page_end is not None and chunk.page_end != chunk.page_start:
            return f"第 {chunk.page_start}-{chunk.page_end} 页"
        return f"第 {chunk.page_start} 页"
    return f"片段 {chunk.chunk_index + 1}"


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
    """按当前解析规则重建存量资料的结构、向量与词面索引。"""
    settings = get_settings()
    documents = list(
        await session.scalars(
            select(KnowledgeDocument)
            .where(KnowledgeDocument.status == "ready")
            .order_by(KnowledgeDocument.created_at)
        )
    )
    rebuilt_chunks = 0
    for document in documents:
        try:
            raw_content = await asyncio.to_thread(Path(document.file_path).read_bytes)
            parsed = await asyncio.to_thread(
                parse_document, document.original_filename, raw_content
            )
            parsed = await apply_semantic_boundaries(
                parsed,
                settings.knowledge_chunk_semantic_similarity_threshold,
                lambda texts: embed_texts(texts, settings),
                settings.knowledge_index_batch_size,
            )
            embeddings = await embed_texts(
                [
                    build_knowledge_retrieval_text(
                        document.title,
                        chunk.heading_path,
                        chunk.content_type,
                        chunk.content,
                    )
                    for chunk in parsed.chunks
                ],
                settings,
            )
            if embeddings is None:
                raise RuntimeError("本地嵌入模型不可用")
            await session.execute(
                delete(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id)
            )
            session.add_all(
                KnowledgeChunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=chunk.content,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    heading=chunk.heading,
                    heading_path=chunk.heading_path,
                    content_type=chunk.content_type,
                    embedding_vector=embedding,
                    search_terms=build_knowledge_search_terms(
                        document.title,
                        chunk.heading_path,
                        chunk.content_type,
                        chunk.content,
                    ),
                    embedding_model=settings.local_embedding_model_id,
                )
                for index, (chunk, embedding) in enumerate(
                    zip(parsed.chunks, embeddings, strict=True)
                )
            )
            document.content = parsed.content
            document.chunk_count = len(parsed.chunks)
            document.error_message = None
            rebuilt_chunks += len(parsed.chunks)
        except Exception:
            logger.exception("knowledge_chunk_reindex_failed document_id=%s", document.id)
    await session.commit()
    return rebuilt_chunks
