"""验证知识库候选召回在数据库边界内完成，应用层仅融合少量 ID。"""

import pytest

from app.services.knowledge import retrieval as knowledge_rag
from app.services.knowledge.search import build_knowledge_search_terms


def test_search_terms_include_document_metadata_and_content() -> None:
    """标题和层级词应进入词面索引，避免仅正文参与召回而漏掉活动名称。"""
    terms = build_knowledge_search_terms(
        "618 夏日焕新活动规则", "优惠玩法 > 店铺券", "paragraph", "店铺券满 199 元减 20 元"
    )

    assert "618" in terms
    assert "优惠" in terms
    assert "店铺" in terms


def test_tsquery_text_only_contains_limited_safe_lexemes() -> None:
    """用户输入不会直接拼入 PostgreSQL tsquery 语法。"""
    tsquery = knowledge_rag._build_tsquery_text("618 优惠券 | !删除 <script>")

    assert tsquery is not None
    assert "|" in tsquery
    assert "script" in tsquery
    assert "<" not in tsquery
    assert "!" not in tsquery


@pytest.mark.asyncio
async def test_database_candidate_rankings_are_rrf_fused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """每条扩展 Query 均分别发起向量和词面召回，再由 RRF 按 ID 去重。"""
    dense_calls: list[list[float]] = []
    sparse_calls: list[str] = []

    async def fake_dense(_session: object, _user_id: str, embedding: list[float]) -> list[str]:
        dense_calls.append(embedding)
        return ["semantic", "shared"]

    async def fake_sparse(_session: object, _user_id: str, query: str) -> list[str]:
        sparse_calls.append(query)
        return ["shared", "keyword"]

    monkeypatch.setattr(knowledge_rag, "_retrieve_dense_candidate_ids", fake_dense)
    monkeypatch.setattr(knowledge_rag, "_retrieve_sparse_candidate_ids", fake_sparse)

    result = await knowledge_rag.retrieve_knowledge_candidates(
        session=object(),  # type: ignore[arg-type]
        user_id="user-1",
        queries=("618 优惠券", "618 发券规则"),
        query_embeddings=([0.1, 0.2], None),
    )

    assert dense_calls == [[0.1, 0.2]]
    assert sparse_calls == ["618 优惠券", "618 发券规则"]
    assert result[0].item_id == "shared"
    assert result[0].hit_count == 3
