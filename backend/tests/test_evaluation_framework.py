"""验证评测数据格式、指标计算和质量门禁不依赖模型或数据库。"""

import json
from pathlib import Path

import pytest

from app.evaluation.dataset import EvaluationCase, EvidenceTarget, load_dataset
from app.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from app.evaluation.runner import (
    ChunkMetadata,
    METRIC_DEFINITIONS,
    _baseline_regression_failures,
    _evidence_targets_match,
    _quality_gate_failures,
    _render_markdown,
)


def test_retrieval_metrics_calculate_expected_scores() -> None:
    ranking = ("noise", "gold-a", "gold-b")
    relevant = ("gold-a", "gold-b")

    assert recall_at_k(ranking, relevant, 2) == 0.5
    assert recall_at_k(ranking, relevant, 3) == 1.0
    assert precision_at_k(ranking, relevant, 2) == 0.5
    assert reciprocal_rank(ranking, relevant) == 0.5
    assert 0 < ndcg_at_k(ranking, relevant, 3) < 1


def test_dataset_loads_versioned_cases() -> None:
    root = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "v1"
    dataset = load_dataset(root)

    assert dataset.version == "v1"
    assert len(dataset.cases) >= 10
    assert any(case.context_turns for case in dataset.cases)
    assert any(case.expected_knowledge_empty for case in dataset.cases)


def test_quality_gates_and_baseline_regression_are_reported(tmp_path) -> None:
    metrics = {"route_accuracy": 0.7, "knowledge_final_recall_at_4": 0.8}
    assert _quality_gate_failures(metrics, {"route_accuracy": 0.8}) == [
        "route_accuracy=0.7000 低于最低门槛 0.8000"
    ]

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"max_regression": 0.03, "metrics": {"knowledge_final_recall_at_4": 0.9}}),
        encoding="utf-8",
    )
    failures = _baseline_regression_failures(baseline_path, metrics)
    assert len(failures) == 1
    assert "knowledge_final_recall_at_4" in failures[0]


def test_invalid_dataset_fails_fast(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text('{"version":"v1","documents":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="没有样例"):
        load_dataset(tmp_path)


def test_evidence_target_checks_document_heading_and_anchor() -> None:
    case = EvaluationCase(
        case_id="anchor",
        category="knowledge",
        question="规则是什么？",
        expected_route="knowledge",
        expected_tools=("query_knowledge_rag",),
        evidence=(EvidenceTarget("rules.md", "优惠规则", "可以叠加"),),
    )
    chunks = {
        "chunk-1": ChunkMetadata("rules.md", "活动规则 > 优惠规则", "优惠可以叠加使用"),
    }

    assert _evidence_targets_match(case, ("chunk-1",), chunks) is True
    assert _evidence_targets_match(case, (), chunks) is False


def test_report_renders_human_readable_metric_definitions() -> None:
    report = {
        "dataset_version": "test",
        "generated_at": "2026-09-14T00:00:00+00:00",
        "models": {"embedding": "test-model", "reranker": None},
        "metrics": {"route_accuracy": 1.0, "latency_p95_ms": 120.0},
        "metric_definitions": METRIC_DEFINITIONS,
        "passed": True,
        "gate_failures": [],
        "baseline_failures": [],
        "cases": [
            {
                "case_id": "routing-failure",
                "route_ok": False,
                "plan_ok": False,
                "metric_ok": True,
                "evidence_ok": True,
                "empty_ok": True,
                "route_decision_reason": "intent_reranker:below_minimum_margin",
            }
        ],
    }

    markdown = _render_markdown(report)
    assert "路由准确率" in markdown
    assert "问题被正确判断" in markdown
    assert "越低越好" in markdown
    assert "intent_reranker:below_minimum_margin" in markdown
