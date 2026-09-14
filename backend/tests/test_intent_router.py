import pytest

import app.services.intent_router as intent_router
from app.config import Settings
from app.services.intent_router import choose_retrieval_route, choose_retrieval_route_with_reason


def test_router_selects_metrics_for_a_confident_metric_question() -> None:
    mode, confidence, fallback = choose_retrieval_route(
        {"metrics": 0.88, "knowledge": 0.31, "hybrid": 0.42}
    )

    assert mode == "metrics"
    assert confidence == 0.88
    assert fallback is False


def test_router_selects_knowledge_for_a_confident_document_question() -> None:
    mode, _, fallback = choose_retrieval_route(
        {"metrics": 0.23, "knowledge": 0.76, "hybrid": 0.48}
    )

    assert mode == "knowledge"
    assert fallback is False


def test_router_falls_back_to_hybrid_when_scores_are_too_close() -> None:
    mode, _, fallback = choose_retrieval_route(
        {"metrics": 0.72, "knowledge": 0.67, "hybrid": 0.59}
    )

    assert mode == "hybrid"
    assert fallback is True


def test_router_selects_web_for_a_confident_current_external_question() -> None:
    mode, confidence, fallback = choose_retrieval_route(
        {"metrics": 0.21, "knowledge": 0.32, "hybrid": 0.37, "web": 0.81}
    )

    assert mode == "web"
    assert confidence == 0.81
    assert fallback is False


def test_router_selects_web_hybrid_for_a_confident_combined_question() -> None:
    mode, confidence, fallback = choose_retrieval_route(
        {"metrics": 0.25, "knowledge": 0.34, "hybrid": 0.55, "web": 0.49, "web_hybrid": 0.84}
    )

    assert mode == "web_hybrid"
    assert confidence == 0.84
    assert fallback is False


def test_router_marks_a_real_hybrid_winner_as_non_fallback() -> None:
    mode, confidence, fallback, reason = choose_retrieval_route_with_reason(
        {"metrics": 0.33, "knowledge": 0.48, "hybrid": 0.78}
    )

    assert (mode, confidence, fallback, reason) == ("hybrid", 0.78, False, "hybrid_selected")


@pytest.mark.asyncio
async def test_router_uses_reranker_to_disambiguate_a_close_vector_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_embed(_: list[str], __: Settings) -> list[list[float]]:
        return [[1.0, 0.0]]

    async def fake_prototypes(_: Settings) -> dict[str, tuple[list[float], ...]]:
        return {mode: ([1.0, 0.0],) for mode in intent_router.INTENT_PROTOTYPES}

    async def fake_rerank(_: str, __: list[str], ___: Settings) -> list[float]:
        return [0.12, 0.91, 0.18, 0.04, 0.03]

    monkeypatch.setattr(intent_router, "embed_texts", fake_embed)
    monkeypatch.setattr(intent_router, "_get_prototype_embeddings", fake_prototypes)
    monkeypatch.setattr(intent_router, "rerank_texts", fake_rerank)

    route = await intent_router.route_question("618 的发货时效要求是什么？", Settings())

    assert route.mode == "knowledge"
    assert route.fallback_to_hybrid is False
    assert route.decision_reason == "intent_reranker:confident_single_route"
    assert route.candidate_scores == {
        "metrics": 0.12,
        "knowledge": 0.91,
        "hybrid": 0.18,
        "web": 0.04,
        "web_hybrid": 0.03,
    }
