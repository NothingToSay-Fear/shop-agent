"""向量、BM25 与 RRF 融合的无状态混合召回能力。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TypeVar

import jieba
from rank_bm25 import BM25Okapi


Item = TypeVar("Item")
DEFAULT_RETRIEVAL_LIMIT = 20
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedCandidate:
    """RRF 融合后的候选项；同一标识只会保留一项。"""

    item_id: str
    rrf_score: float
    hit_count: int


def tokenize_for_bm25(text: str) -> list[str]:
    """使用中文分词生成 BM25 词项，不使用 n-gram 作为检索兜底。"""
    return [token.strip().lower() for token in jieba.lcut(text) if token.strip()]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], rrf_k: int = DEFAULT_RRF_K
) -> list[FusedCandidate]:
    """按名次而非不同检索器的原始分数执行 RRF，并在融合时去重。"""
    scores: dict[str, float] = {}
    hit_counts: dict[str, int] = {}
    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for rank, item_id in enumerate(ranking, start=1):
            if item_id in seen_in_ranking:
                continue
            seen_in_ranking.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (rrf_k + rank)
            hit_counts[item_id] = hit_counts.get(item_id, 0) + 1
    return sorted(
        (
            FusedCandidate(item_id=item_id, rrf_score=score, hit_count=hit_counts[item_id])
            for item_id, score in scores.items()
        ),
        key=lambda item: (-item.rrf_score, -item.hit_count, item.item_id),
    )


def hybrid_retrieve(
    items: Sequence[Item],
    queries: Sequence[str],
    query_embeddings: Sequence[list[float] | None],
    *,
    get_id: Callable[[Item], str],
    get_text: Callable[[Item], str],
    get_embedding: Callable[[Item], list[float] | None],
    dense_score: Callable[[Item, str, list[float]], float],
    retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
    rrf_k: int = DEFAULT_RRF_K,
    dense_minimum_score: float = 0.2,
) -> list[FusedCandidate]:
    """对每个 Query 分别进行向量和 BM25 召回，再以 RRF 融合结果。"""
    if not items or not queries or retrieval_limit <= 0:
        return []

    rankings: list[list[str]] = []
    for query, query_embedding in zip(queries, query_embeddings, strict=True):
        if query_embedding:
            dense = _dense_ranking(
                items,
                query,
                query_embedding,
                get_id,
                get_embedding,
                dense_score,
                retrieval_limit,
                dense_minimum_score,
            )
            if dense:
                rankings.append(dense)
        sparse = _bm25_ranking(items, query, get_id, get_text, retrieval_limit)
        if sparse:
            rankings.append(sparse)
    return reciprocal_rank_fusion(rankings, rrf_k)


def _dense_ranking(
    items: Sequence[Item],
    query: str,
    query_embedding: list[float],
    get_id: Callable[[Item], str],
    get_embedding: Callable[[Item], list[float] | None],
    dense_score: Callable[[Item, str, list[float]], float],
    retrieval_limit: int,
    minimum_score: float,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for item in items:
        embedding = get_embedding(item)
        if embedding is None or len(embedding) != len(query_embedding):
            continue
        score = dense_score(item, query, query_embedding)
        if isfinite(score) and score >= minimum_score:
            scored.append((score, get_id(item)))
    return [item_id for _, item_id in sorted(scored, reverse=True)[:retrieval_limit]]


def _bm25_ranking(
    items: Sequence[Item],
    query: str,
    get_id: Callable[[Item], str],
    get_text: Callable[[Item], str],
    retrieval_limit: int,
) -> list[str]:
    query_tokens = tokenize_for_bm25(query)
    corpus = [tokenize_for_bm25(get_text(item)) for item in items]
    if not query_tokens or not any(corpus):
        return []
    scores = BM25Okapi(corpus).get_scores(query_tokens)
    scored = [(float(score), get_id(item)) for item, score in zip(items, scores, strict=True) if score > 0]
    return [item_id for _, item_id in sorted(scored, reverse=True)[:retrieval_limit]]
