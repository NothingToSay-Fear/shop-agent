from app.services.intent_router import choose_retrieval_route


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
