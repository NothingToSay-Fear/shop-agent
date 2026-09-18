from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.services.retrieval.hybrid import hybrid_retrieve, reciprocal_rank_fusion, tokenize_for_bm25
from app.services.knowledge import retrieval as knowledge_rag
from app.config import Settings
from app.services.retrieval.hybrid import FusedCandidate
from app.services.retrieval.query_expansion import _normalize_expanded_queries


@dataclass(frozen=True)
class RetrievalItem:
    """为混合召回单元测试准备的最小候选对象。"""

    item_id: str
    text: str
    embedding: list[float]


def test_rrf_fuses_rankings_and_deduplicates_items() -> None:
    """同一候选被多路命中时应累加名次分，并且只在最终结果中出现一次。"""
    result = reciprocal_rank_fusion([["a", "shared"], ["shared", "b"], ["shared"]])

    assert result[0].item_id == "shared"
    assert result[0].hit_count == 3
    assert [item.item_id for item in result].count("shared") == 1


def test_hybrid_retrieval_combines_dense_and_bm25_candidates() -> None:
    """关键词命中与语义命中均应进入 RRF，而不是互相覆盖。"""
    items = [
        RetrievalItem("coupon", "618 店铺券满 199 元减 20 元，可叠加跨店满减", [1.0, 0.0]),
        RetrievalItem("shipping", "618 现货订单支付后 48 小时内发货", [0.0, 1.0]),
        # 保留无关片段，使仅在优惠券片段出现的关键词拥有非零 BM25 IDF。
        RetrievalItem("other", "春季上新需要补充商品尺寸对比图", [0.0, 0.0]),
    ]
    result = hybrid_retrieve(
        items,
        ("618 优惠券可以叠加吗",),
        [[1.0, 0.0]],
        get_id=lambda item: item.item_id,
        get_text=lambda item: item.text,
        get_embedding=lambda item: item.embedding,
        dense_score=lambda item, _query, vector: sum(
            left * right for left, right in zip(item.embedding, vector, strict=True)
        ),
    )

    assert result[0].item_id == "coupon"
    assert result[0].hit_count >= 2
    assert "优惠券" in tokenize_for_bm25("618 优惠券可以叠加吗")


def test_query_expansion_keeps_original_query_and_limits_untrusted_output() -> None:
    """模型扩展输出只能追加受限改写，不能替换原问题或无限增长。"""
    result = _normalize_expanded_queries(
        "618 怎么发券",
        '{"queries":["618 优惠券发放规则", "618 券领取条件", "618 券使用限制", 3]}',
        2,
    )

    assert result == ("618 怎么发券", "618 优惠券发放规则", "618 券领取条件")


@pytest.mark.asyncio
async def test_reranker_filters_candidates_below_configured_minimum_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """精排已成功运行时，低于阈值的候选不能作为知识库依据返回。"""
    relevant_chunk = SimpleNamespace(id="relevant", heading=None, content="618 店铺券可以叠加")
    irrelevant_chunk = SimpleNamespace(id="irrelevant", heading=None, content="七夕礼盒库存不足")
    document = SimpleNamespace(title="活动资料")

    async def fake_rerank(*_args: object, **_kwargs: object) -> list[float]:
        return [0.82, 0.12]

    monkeypatch.setattr(knowledge_rag, "rerank_texts", fake_rerank)
    result = await knowledge_rag._rerank_candidates(
        "618 优惠券能叠加吗",
        [
            FusedCandidate("relevant", 0.03, 2),
            FusedCandidate("irrelevant", 0.02, 1),
        ],
        [(relevant_chunk, document), (irrelevant_chunk, document)],
        Settings(knowledge_reranker_min_score=0.35),
    )

    assert result == [(relevant_chunk, document)]
