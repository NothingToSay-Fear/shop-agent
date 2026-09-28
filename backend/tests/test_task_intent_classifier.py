import pytest

import app.services.conversations.task_intent_classifier as classifier
from app.config import Settings


@pytest.mark.asyncio
async def test_vector_prototypes_select_a_clear_task_type(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_embed(_: list[str], __: Settings) -> list[list[float]]:
        return [[1.0, 0.0]]

    async def fake_prototypes(_: Settings) -> dict[str, tuple[list[float], ...]]:
        return {
            "metric_query": ([0.91, 0.0],),
            "metric_comparison": ([0.60, 0.0],),
            "knowledge_qa": ([0.20, 0.0],),
            "hybrid_analysis": ([0.30, 0.0],),
            "review": ([0.10, 0.0],),
        }

    monkeypatch.setattr(classifier, "embed_texts", fake_embed)
    monkeypatch.setattr(classifier, "_get_prototype_embeddings", fake_prototypes)

    result = await classifier.classify_task_type(
        "给我看看上周 GMV", "knowledge_qa", Settings()
    )

    assert result.task_type == "metric_query"
    assert result.fallback_to_rule is False
    assert result.decision_reason == "vector_confident"
    assert result.candidate_scores == {
        "metric_query": 0.91,
        "metric_comparison": 0.6,
        "knowledge_qa": 0.2,
        "hybrid_analysis": 0.3,
        "review": 0.1,
    }


@pytest.mark.asyncio
async def test_close_vector_scores_fall_back_to_the_deterministic_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_embed(_: list[str], __: Settings) -> list[list[float]]:
        return [[1.0, 0.0]]

    async def fake_prototypes(_: Settings) -> dict[str, tuple[list[float], ...]]:
        return {
            "metric_query": ([0.72, 0.0],),
            "metric_comparison": ([0.69, 0.0],),
            "knowledge_qa": ([0.20, 0.0],),
            "hybrid_analysis": ([0.30, 0.0],),
            "review": ([0.10, 0.0],),
        }

    monkeypatch.setattr(classifier, "embed_texts", fake_embed)
    monkeypatch.setattr(classifier, "_get_prototype_embeddings", fake_prototypes)

    result = await classifier.classify_task_type("含糊问题", "knowledge_qa", Settings())

    assert result.task_type == "knowledge_qa"
    assert result.fallback_to_rule is True
    assert result.decision_reason == "rule_fallback:below_minimum_margin"
