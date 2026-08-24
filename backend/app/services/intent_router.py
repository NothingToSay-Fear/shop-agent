"""以本地语义向量决定应检索哪些 RAG 数据源。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from app.config import Settings, get_settings
from app.services.local_embeddings import embed_texts
from app.services.metric_rag import cosine_similarity

RetrievalMode = Literal["metrics", "knowledge", "hybrid", "web"]
RouteMode = Literal["metrics", "knowledge", "hybrid", "web", "web_hybrid"]

# 原型描述是稳定的产品语义，不是针对某个业务词的硬编码分支。
INTENT_PROTOTYPES: dict[RouteMode, str] = {
    "metrics": "查询经营数据、GMV、订单数、访客数、转化率、客单价、退款率、趋势、同比、环比和数值表现。",
    "knowledge": "查询活动规则、玩法说明、运营手册、商品资料、历史方案、活动复盘、流程和已有文档中的事实。",
    "hybrid": "结合经营数据和活动规则或历史资料进行归因、效果分析、优化建议、复盘和下一步运营动作。",
    "web": "查询最新的公开互联网信息、平台政策、行业动态、竞品新闻、市场趋势、实时资讯和外部资料来源。",
    "web_hybrid": "结合内部经营数据、活动资料与最新公开平台政策、行业动态或竞品信息，完成综合分析和建议。",
}
# 以当前 bge-small-zh-v1.5 的实际中文问题分数校准；仍以分差保护综合问题。
MINIMUM_CONFIDENCE = 0.45
MINIMUM_MARGIN = 0.06


@dataclass(frozen=True)
class RetrievalRoute:
    """路由结论与可复用的问题向量。"""

    mode: RouteMode
    query_embedding: list[float] | None
    confidence: float
    fallback_to_hybrid: bool


_prototype_embedding_cache: dict[str, dict[RouteMode, list[float]]] = {}
_prototype_embedding_lock = asyncio.Lock()


def choose_retrieval_route(scores: dict[RouteMode, float]) -> tuple[RouteMode, float, bool]:
    """按阈值和第一、二名分差选择路由；不确定时保守地走综合检索。"""
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, confidence = ranked[0]
    runner_up_score = ranked[1][1]
    if confidence < MINIMUM_CONFIDENCE or confidence - runner_up_score < MINIMUM_MARGIN:
        return "hybrid", confidence, True
    return winner, confidence, False


async def route_question(
    question: str, settings: Settings | None = None
) -> RetrievalRoute:
    """只生成一次问题向量，并与缓存的三类意图原型进行余弦相似度比较。"""
    active_settings = settings or get_settings()
    question_vectors = await embed_texts([question], active_settings)
    if not question_vectors:
        # 本地模型不可用时保留原有综合检索兜底，具体检索层会返回无上下文。
        return RetrievalRoute("hybrid", None, 0.0, True)

    prototype_embeddings = await _get_prototype_embeddings(active_settings)
    if prototype_embeddings is None:
        return RetrievalRoute("hybrid", question_vectors[0], 0.0, True)

    scores = {
        mode: cosine_similarity(question_vectors[0], prototype_embeddings[mode])
        for mode in INTENT_PROTOTYPES
    }
    mode, confidence, fallback_to_hybrid = choose_retrieval_route(scores)
    return RetrievalRoute(mode, question_vectors[0], confidence, fallback_to_hybrid)


async def _get_prototype_embeddings(
    settings: Settings,
) -> dict[RouteMode, list[float]] | None:
    """按模型版本和设备缓存意图原型向量，避免每个问题重复向量化原型文本。"""
    cache_key = f"{settings.local_embedding_model_id}:{settings.local_embedding_model_path}:{settings.local_embedding_device}"
    cached = _prototype_embedding_cache.get(cache_key)
    if cached is not None:
        return cached

    async with _prototype_embedding_lock:
        cached = _prototype_embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        vectors = await embed_texts(list(INTENT_PROTOTYPES.values()), settings)
        if vectors is None:
            return None
        cached = dict(zip(INTENT_PROTOTYPES, vectors, strict=True))
        _prototype_embedding_cache[cache_key] = cached
        return cached
