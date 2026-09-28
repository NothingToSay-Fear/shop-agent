"""使用意图原型向量识别结构化任务类型，并保留可审计的规则降级。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from app.config import Settings, get_settings
from app.services.models.embeddings import embed_texts

TaskType = Literal[
    "metric_query",
    "metric_comparison",
    "knowledge_qa",
    "hybrid_analysis",
    "review",
]

# 每个类别使用多条简短、正向且互斥的代表问法；分类时取类别内最高相似度。
TASK_TYPE_VECTOR_PROTOTYPES: dict[TaskType, tuple[str, ...]] = {
    "metric_query": (
        "查询某个日期或活动的 GMV、订单量、访客数、转化率等经营指标数值",
        "查看销售额、客单价、退款率或订单量的当前数据和趋势",
    ),
    "metric_comparison": (
        "比较两个时间段、活动或渠道的 GMV、订单量、转化率等经营指标",
        "分析经营数据的同比、环比、差异和涨跌幅",
    ),
    "knowledge_qa": (
        "查询内部资料中的活动规则、商品资料、发货时效、售后流程或玩法说明",
        "从文档、手册或历史资料中查找某项规定和事实",
    ),
    "hybrid_analysis": (
        "根据经营指标数据分析变化原因，并结合资料给出优化建议",
        "对 GMV、转化率或订单变化做归因诊断和经营分析",
    ),
    "review": (
        "复盘某个活动的经营效果、执行动作、问题和后续改进计划",
        "评估活动效果，汇总数据证据和经验，输出完整复盘结论",
    ),
}

TASK_TYPE_MINIMUM_CONFIDENCE = 0.45
TASK_TYPE_MINIMUM_MARGIN = 0.06


@dataclass(frozen=True)
class TaskTypeClassification:
    """任务类型判定及其审计信息。"""

    task_type: TaskType
    confidence: float
    fallback_to_rule: bool
    decision_reason: str
    candidate_scores: dict[str, float] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "task_type": self.task_type,
            "confidence": round(self.confidence, 4),
            "fallback_to_rule": self.fallback_to_rule,
            "decision_reason": self.decision_reason,
            "candidate_scores": self.candidate_scores,
        }


_prototype_embedding_cache: dict[str, dict[TaskType, tuple[list[float], ...]]] = {}
_prototype_embedding_lock = asyncio.Lock()


async def classify_task_type(
    question: str,
    rule_fallback: TaskType,
    settings: Settings | None = None,
) -> TaskTypeClassification:
    """向量分类不确定或不可用时，返回既有确定性规则的安全结论。"""
    active_settings = settings or get_settings()
    question_vectors = await embed_texts([question], active_settings)
    if not question_vectors:
        return _rule_fallback(rule_fallback, "vector_unavailable")

    prototype_embeddings = await _get_prototype_embeddings(active_settings)
    if prototype_embeddings is None:
        return _rule_fallback(rule_fallback, "prototype_vector_unavailable")

    vector_scores = {
        task_type: max(
            _cosine_similarity(question_vectors[0], prototype_embedding)
            for prototype_embedding in prototype_embeddings[task_type]
        )
        for task_type in TASK_TYPE_VECTOR_PROTOTYPES
    }
    ranked = sorted(vector_scores.items(), key=lambda item: item[1], reverse=True)
    winner, confidence = ranked[0]
    runner_up_score = ranked[1][1]
    scores = {name: round(score, 4) for name, score in vector_scores.items()}
    if confidence < TASK_TYPE_MINIMUM_CONFIDENCE:
        return _rule_fallback(rule_fallback, "below_minimum_confidence", confidence, scores)
    if confidence - runner_up_score < TASK_TYPE_MINIMUM_MARGIN:
        return _rule_fallback(rule_fallback, "below_minimum_margin", confidence, scores)
    return TaskTypeClassification(winner, confidence, False, "vector_confident", scores)


def _rule_fallback(
    task_type: TaskType,
    reason: str,
    confidence: float = 0.0,
    candidate_scores: dict[str, float] | None = None,
) -> TaskTypeClassification:
    return TaskTypeClassification(task_type, confidence, True, f"rule_fallback:{reason}", candidate_scores)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """嵌入已归一化；长度异常时不把它当作有效语义命中。"""
    if len(left) != len(right) or not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


async def _get_prototype_embeddings(
    settings: Settings,
) -> dict[TaskType, tuple[list[float], ...]] | None:
    """按本地模型配置缓存模板向量，避免每轮任务重复编码。"""
    cache_key = (
        f"{settings.local_embedding_model_id}:"
        f"{settings.local_embedding_model_path}:"
        f"{settings.local_embedding_device}"
    )
    cached = _prototype_embedding_cache.get(cache_key)
    if cached is not None:
        return cached

    async with _prototype_embedding_lock:
        cached = _prototype_embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        flat_prototypes = [
            prototype
            for prototypes in TASK_TYPE_VECTOR_PROTOTYPES.values()
            for prototype in prototypes
        ]
        vectors = await embed_texts(flat_prototypes, settings)
        if vectors is None or len(vectors) != len(flat_prototypes):
            return None
        cached: dict[TaskType, tuple[list[float], ...]] = {}
        offset = 0
        for task_type, prototypes in TASK_TYPE_VECTOR_PROTOTYPES.items():
            cached[task_type] = tuple(vectors[offset : offset + len(prototypes)])
            offset += len(prototypes)
        _prototype_embedding_cache[cache_key] = cached
        return cached
