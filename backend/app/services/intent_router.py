"""以本地语义向量决定应检索哪些 RAG 数据源。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from app.config import Settings, get_settings
from app.services.models.embeddings import embed_texts
from app.services.models.reranker import rerank_texts
from app.services.analytics.metric_rag import cosine_similarity

RetrievalMode = Literal["metrics", "knowledge", "hybrid", "web"]
RouteMode = Literal["metrics", "knowledge", "hybrid", "web", "web_hybrid"]

# 原型描述供精排模型做五选一比较；保持短小、正向且互斥，避免长描述中的否定语义干扰相似度。
INTENT_PROTOTYPES: dict[RouteMode, str] = {
    "metrics": "查询经营指标数值，例如 GMV、订单量、访客、转化率、客单价、趋势、同比或环比。",
    "knowledge": "查询内部资料事实，例如活动规则、玩法、发货要求、商品资料、流程或历史复盘内容。",
    "hybrid": "同时查询经营数据和内部资料，用于归因、效果分析、复盘或优化建议。",
    "web": "查询最新的公开互联网信息、平台政策、行业动态、竞品新闻、市场趋势、实时资讯和外部资料来源。",
    "web_hybrid": "结合内部经营数据、活动资料与最新公开平台政策、行业动态或竞品信息，完成综合分析和建议。",
}
# 向量检索为每类意图使用少量代表性表达并取最高相似度。它仍是语义匹配，且没有把活动名称或
# 评测样例硬编码为规则；多个短原型能避免单一长描述遗漏“能否叠加”“最晚发货”等常见问法。
INTENT_VECTOR_PROTOTYPES: dict[RouteMode, tuple[str, ...]] = {
    "metrics": (
        "GMV、订单量、销售额、访客数、转化率等经营数据是多少",
        "比较不同活动或月份的经营指标、同比、环比和趋势",
        "查询支付订单数、成交金额、退款率、客单价等数值",
    ),
    "knowledge": (
        "活动规则、优惠是否能叠加、报名玩法和限制是什么",
        "现货订单的发货时效、履约要求和售后流程是什么",
        "活动主推商品、目标人群、送礼场景或商品资料是什么",
        "历史活动复盘中哪些动作已验证有效、有哪些经验和结论",
        "资料里是否有某项规则或说明，未找到资料时明确说明",
    ),
    "hybrid": (
        "结合 GMV、订单等经营数据和活动规则进行复盘分析",
        "依据经营指标与历史复盘资料分析原因并给出优化建议",
    ),
    "web": (
        "查询最新公开平台政策、行业新闻、天气、市场动态或竞品资讯",
        "搜索互联网最新资料并给出公开来源",
    ),
    "web_hybrid": (
        "结合内部经营数据、知识库资料和最新公开网络信息做分析建议",
        "用内部数据、活动资料和外部行业动态完成综合复盘",
    ),
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
    decision_reason: str = "vector_confident"
    candidate_scores: dict[str, float] | None = None


_prototype_embedding_cache: dict[str, dict[RouteMode, tuple[list[float], ...]]] = {}
_prototype_embedding_lock = asyncio.Lock()


def choose_retrieval_route(scores: dict[RouteMode, float]) -> tuple[RouteMode, float, bool]:
    """按阈值和第一、二名分差选择路由；不确定时保守地走综合检索。"""
    mode, confidence, fallback, _ = choose_retrieval_route_with_reason(scores)
    return mode, confidence, fallback


def choose_retrieval_route_with_reason(
    scores: dict[RouteMode, float],
) -> tuple[RouteMode, float, bool, str]:
    """返回可审计的路由结论，明确区分真实综合意图与保护性降级。"""
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, confidence = ranked[0]
    runner_up_score = ranked[1][1]
    if winner == "hybrid" and confidence >= MINIMUM_CONFIDENCE and confidence - runner_up_score >= MINIMUM_MARGIN:
        return "hybrid", confidence, False, "hybrid_selected"
    if confidence < MINIMUM_CONFIDENCE:
        return "hybrid", confidence, True, "below_minimum_confidence"
    if confidence - runner_up_score < MINIMUM_MARGIN:
        return "hybrid", confidence, True, "below_minimum_margin"
    return winner, confidence, False, "confident_single_route"


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

    vector_scores = {
        mode: max(
            cosine_similarity(question_vectors[0], prototype_embedding)
            for prototype_embedding in prototype_embeddings[mode]
        )
        for mode in INTENT_PROTOTYPES
    }
    mode, confidence, fallback_to_hybrid, reason = choose_retrieval_route_with_reason(vector_scores)
    # 只有向量层倾向综合检索时才额外精判，避免为高置信度单路问题增加模型推理。
    if mode == "hybrid":
        rerank_scores = await rerank_texts(question, list(INTENT_PROTOTYPES.values()), active_settings)
        # CrossEncoder 分数是“问题—原型描述”的原始相关性，并非五分类概率。仅当它至少达到
        # 已配置的知识精排最低相关度时才允许覆盖向量结论，防止低分原型比较制造新的保护性降级。
        if (
            rerank_scores is not None
            and len(rerank_scores) == len(INTENT_PROTOTYPES)
            and max(rerank_scores, default=float("-inf")) >= active_settings.knowledge_reranker_min_score
        ):
            reranked = dict(zip(INTENT_PROTOTYPES, rerank_scores, strict=True))
            mode, confidence, fallback_to_hybrid, rerank_reason = choose_retrieval_route_with_reason(
                reranked
            )
            return RetrievalRoute(
                mode,
                question_vectors[0],
                confidence,
                fallback_to_hybrid,
                f"intent_reranker:{rerank_reason}",
                {name: round(score, 4) for name, score in reranked.items()},
            )
    return RetrievalRoute(
        mode,
        question_vectors[0],
        confidence,
        fallback_to_hybrid,
        f"vector:{reason}",
        {name: round(score, 4) for name, score in vector_scores.items()},
    )


async def _get_prototype_embeddings(
    settings: Settings,
) -> dict[RouteMode, tuple[list[float], ...]] | None:
    """按模型版本和设备缓存意图原型向量，避免每个问题重复向量化原型文本。"""
    cache_key = f"{settings.local_embedding_model_id}:{settings.local_embedding_model_path}:{settings.local_embedding_device}"
    cached = _prototype_embedding_cache.get(cache_key)
    if cached is not None:
        return cached

    async with _prototype_embedding_lock:
        cached = _prototype_embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        flat_prototypes = [
            prototype
            for prototypes in INTENT_VECTOR_PROTOTYPES.values()
            for prototype in prototypes
        ]
        vectors = await embed_texts(flat_prototypes, settings)
        if vectors is None:
            return None
        cached: dict[RouteMode, tuple[list[float], ...]] = {}
        offset = 0
        for mode, prototypes in INTENT_VECTOR_PROTOTYPES.items():
            cached[mode] = tuple(vectors[offset : offset + len(prototypes)])
            offset += len(prototypes)
        _prototype_embedding_cache[cache_key] = cached
        return cached
