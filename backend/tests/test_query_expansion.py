import pytest

import app.services.query_expansion as query_expansion
from app.config import Settings


@pytest.mark.asyncio
async def test_prepare_retrieval_queries_reuses_route_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """请求级检索准备应复用路由阶段生成的原问题向量。"""
    original_embedding = [0.1, 0.2]
    calls: list[object] = []

    async def fake_expand(question: str, _: Settings) -> tuple[str, ...]:
        calls.append(("expand", question))
        return (question, "618 活动成交额下降原因")

    async def fake_embed(
        queries: tuple[str, ...], _: Settings, embedding: list[float] | None
    ) -> list[list[float] | None]:
        calls.append(("embed", queries, embedding))
        return [embedding, [0.3, 0.4]]

    monkeypatch.setattr(query_expansion, "expand_queries", fake_expand)
    monkeypatch.setattr(query_expansion, "embed_expanded_queries", fake_embed)

    prepared = await query_expansion.prepare_retrieval_queries(
        "  618 的 GMV 为什么下降？  ", Settings(), original_embedding
    )

    assert prepared.question == "618 的 GMV 为什么下降？"
    assert prepared.queries == ("618 的 GMV 为什么下降？", "618 活动成交额下降原因")
    assert prepared.original_embedding is original_embedding
    assert calls == [
        ("expand", "618 的 GMV 为什么下降？"),
        ("embed", prepared.queries, original_embedding),
    ]
