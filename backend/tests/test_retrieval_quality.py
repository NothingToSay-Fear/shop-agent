"""使用人工标注的小型资料集验证混合召回的离线 Recall@K。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.config import Settings
from app.services.hybrid_retrieval import hybrid_retrieve
from app.services.local_embeddings import embed_texts
from app.services.metric_rag import (
    METRIC_DEFINITION_SEEDS,
    MetricDocument,
    cosine_similarity,
    retrieve_metrics_for_queries,
)


@dataclass(frozen=True)
class KnowledgeEvaluationItem:
    """检索质量评测中的带人工标注资料片段。"""

    item_id: str
    content: str


KNOWLEDGE_ITEMS = (
    KnowledgeEvaluationItem("618-coupon", "618 正式期店铺券满 199 元减 20 元，跨店满减每满 300 元减 40 元，两者可以叠加。"),
    KnowledgeEvaluationItem("618-shipping", "618 现货订单应在支付后 48 小时内发货；库存不足时 24 小时内联系用户退款。"),
    KnowledgeEvaluationItem("qixi-goods", "七夕主推保温杯礼盒与双肩包加保温杯礼盒组合款，面向情侣和职场送礼场景。"),
    KnowledgeEvaluationItem("qixi-risk", "七夕礼盒库存低于 80 套时暂停直播间组合款投放，避免超卖。"),
    KnowledgeEvaluationItem("spring-coupon", "春季上新针对加购未支付用户在 24 小时内发放满 199 减 15 元定向券，带回部分订单。"),
    KnowledgeEvaluationItem("spring-ads", "春季搜索广告保留轻量双肩包和通勤电脑包等高意图词，晚间 20 点至 23 点提高预算。"),
)

KNOWLEDGE_CASES = (
    ("618 的优惠券与跨店满减能否叠加？", "618-coupon"),
    ("618 现货订单最晚多久发货？", "618-shipping"),
    ("七夕组合款适合哪些送礼人群？", "qixi-goods"),
    ("七夕礼盒库存不足时应该如何处理？", "qixi-risk"),
    ("春季活动怎样召回加购后未付款用户？", "spring-coupon"),
    ("春季上新搜索广告应该保留哪些词和时段？", "spring-ads"),
)

METRIC_CASES = (
    ("成交额是多少", "paid_gmv"),
    ("本周 UV 有多少", "visitor_count"),
    ("这次活动客单价表现如何", "average_order_value"),
    ("售后退款率是否偏高", "refund_rate"),
)


def _local_settings() -> Settings:
    """测试在宿主机或 Docker 中均定位到已有的本地嵌入模型。"""
    repository_root = Path(__file__).resolve().parents[2]
    default_path = repository_root / "models" / "bge-small-zh-v1.5"
    return Settings(local_embedding_model_path=str(default_path))


@pytest.mark.asyncio
async def test_hybrid_retrieval_recall_at_3_on_labeled_knowledge_cases() -> None:
    """人工标注资料集中，目标片段必须进入混合召回 Top-3。"""
    settings = _local_settings()
    document_embeddings = await embed_texts([item.content for item in KNOWLEDGE_ITEMS], settings)
    if document_embeddings is None:
        pytest.skip("本地嵌入模型未准备好，无法执行真实向量召回评测")
    item_embeddings = dict(zip((item.item_id for item in KNOWLEDGE_ITEMS), document_embeddings, strict=True))

    hits = 0
    failures: list[str] = []
    for question, expected_id in KNOWLEDGE_CASES:
        query_embeddings = await embed_texts([question], settings)
        assert query_embeddings is not None
        result = hybrid_retrieve(
            KNOWLEDGE_ITEMS,
            (question,),
            query_embeddings,
            get_id=lambda item: item.item_id,
            get_text=lambda item: item.content,
            get_embedding=lambda item: item_embeddings[item.item_id],
            dense_score=lambda item, _query, vector: cosine_similarity(vector, item_embeddings[item.item_id]),
        )
        retrieved_ids = [item.item_id for item in result[:3]]
        if expected_id in retrieved_ids:
            hits += 1
        else:
            failures.append(f"{question} -> {retrieved_ids}，期望 {expected_id}")

    recall_at_3 = hits / len(KNOWLEDGE_CASES)
    print(f"知识库混合召回 Recall@3：{hits}/{len(KNOWLEDGE_CASES)} = {recall_at_3:.1%}")
    assert recall_at_3 == 1.0, "\n".join(failures)


@pytest.mark.asyncio
async def test_metric_hybrid_retrieval_recall_at_2_on_labeled_metric_cases() -> None:
    """指标名称、业务别名和派生指标表达均应进入 Top-2。"""
    settings = _local_settings()
    documents_without_embeddings = [
        MetricDocument(
            metric_code=seed["metric_code"],
            name=seed["name"],
            description=seed["description"],
            aliases=seed["aliases"],
            embedding=None,
        )
        for seed in METRIC_DEFINITION_SEEDS
    ]
    document_embeddings = await embed_texts(
        [item.retrieval_text for item in documents_without_embeddings], settings
    )
    if document_embeddings is None:
        pytest.skip("本地嵌入模型未准备好，无法执行真实向量召回评测")
    documents = [
        MetricDocument(
            metric_code=item.metric_code,
            name=item.name,
            description=item.description,
            aliases=item.aliases,
            embedding=embedding,
        )
        for item, embedding in zip(documents_without_embeddings, document_embeddings, strict=True)
    ]

    hits = 0
    failures: list[str] = []
    for question, expected_code in METRIC_CASES:
        query_embeddings = await embed_texts([question], settings)
        assert query_embeddings is not None
        result = retrieve_metrics_for_queries((question,), documents, query_embeddings)
        retrieved_codes = [item.metric_code for item in result]
        if expected_code in retrieved_codes:
            hits += 1
        else:
            failures.append(f"{question} -> {retrieved_codes}，期望 {expected_code}")

    recall_at_2 = hits / len(METRIC_CASES)
    print(f"指标混合召回 Recall@2：{hits}/{len(METRIC_CASES)} = {recall_at_2:.1%}")
    assert recall_at_2 == 1.0, "\n".join(failures)
