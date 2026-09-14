"""不依赖外部 SaaS 的检索与回答质量指标计算。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import log2


def recall_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """人工标注依据中有多少比例位于 Top-K。"""
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    return len(set(ranking[:k]) & relevant_set) / len(relevant_set)


def precision_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Top-K 中属于人工标注依据的比例。"""
    selected = ranking[:k]
    if not selected:
        return 1.0
    relevant_set = set(relevant)
    return sum(item in relevant_set for item in selected) / len(selected)


def reciprocal_rank(ranking: Sequence[str], relevant: Iterable[str]) -> float:
    """第一个正确依据的倒数排名，即单问题 MRR 分量。"""
    relevant_set = set(relevant)
    for index, item in enumerate(ranking, start=1):
        if item in relevant_set:
            return 1 / index
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """二元相关性 nDCG，兼顾多个正确依据的排序位置。"""
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    dcg = sum(1 / log2(index + 1) for index, item in enumerate(ranking[:k], start=1) if item in relevant_set)
    ideal_count = min(len(relevant_set), k)
    ideal_dcg = sum(1 / log2(index + 1) for index in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 1.0


def mean(values: Iterable[float]) -> float:
    """返回空集合安全的平均值，便于未覆盖某类样例时输出报告。"""
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0
