"""读取版本化的 RAG 评测集，避免将评测样例散落在测试代码中。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvidenceTarget:
    """人工标注的一条知识依据，以文件和锚点而非易变的 Chunk ID 标识。"""

    document: str
    heading_contains: str | None = None
    anchor_text: str | None = None


@dataclass(frozen=True)
class EvaluationCase:
    """可在真实 RAG 链路上重复执行的一条评测样例。"""

    case_id: str
    category: str
    question: str
    expected_route: str
    expected_tools: tuple[str, ...]
    expected_metric_codes: tuple[str, ...] = ()
    evidence: tuple[EvidenceTarget, ...] = ()
    expected_claims: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    expected_knowledge_empty: bool = False
    context_turns: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationDataset:
    """一份带清单、语料与评测问题的数据集版本。"""

    root: Path
    version: str
    documents: tuple[str, ...]
    cases: tuple[EvaluationCase, ...]
    quality_gates: dict[str, float]


def load_dataset(root: Path) -> EvaluationDataset:
    """读取 manifest 与全部 JSONL 样例，并在加载期发现重复 ID。"""
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    documents = tuple(_string_list(manifest.get("documents"), manifest_path, "documents"))
    quality_gates = {
        key: float(value)
        for key, value in _object(manifest.get("quality_gates", {}), manifest_path, "quality_gates").items()
    }
    cases: list[EvaluationCase] = []
    for path in sorted(root.glob("*.jsonl")):
        cases.extend(_load_cases(path))
    if not cases:
        raise ValueError(f"评测集没有样例：{root}")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"评测集存在重复 case_id：{root}")
    return EvaluationDataset(
        root=root,
        version=str(manifest.get("version", "unknown")),
        documents=documents,
        cases=tuple(cases),
        quality_gates=quality_gates,
    )


def _load_cases(path: Path) -> list[EvaluationCase]:
    result: list[EvaluationCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} 不是合法 JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
        evidence = tuple(
            EvidenceTarget(
                document=_required_string(item, path, line_number, "document"),
                heading_contains=_optional_string(item.get("heading_contains"), path, line_number),
                anchor_text=_optional_string(item.get("anchor_text"), path, line_number),
            )
            for item in _object_list(payload.get("evidence", []), path, line_number, "evidence")
        )
        result.append(
            EvaluationCase(
                case_id=_required_string(payload, path, line_number, "id"),
                category=_required_string(payload, path, line_number, "category"),
                question=_required_string(payload, path, line_number, "question"),
                expected_route=_required_string(payload, path, line_number, "expected_route"),
                expected_tools=tuple(_string_list(payload.get("expected_tools", []), path, "expected_tools")),
                expected_metric_codes=tuple(
                    _string_list(payload.get("expected_metric_codes", []), path, "expected_metric_codes")
                ),
                evidence=evidence,
                expected_claims=tuple(_string_list(payload.get("expected_claims", []), path, "expected_claims")),
                forbidden_claims=tuple(_string_list(payload.get("forbidden_claims", []), path, "forbidden_claims")),
                expected_knowledge_empty=bool(payload.get("expected_knowledge_empty", False)),
                context_turns=tuple(_string_list(payload.get("context_turns", []), path, "context_turns")),
            )
        )
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"缺少评测集清单：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"评测集清单不是合法 JSON：{path}") from error
    return _object(payload, path, "manifest")


def _object(value: Any, path: Path, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} 的 {field} 必须是对象")
    return value


def _object_list(value: Any, path: Path, line: int, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path}:{line} 的 {field} 必须是对象数组")
    return value


def _string_list(value: Any, path: Path, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{path} 的 {field} 必须是非空字符串数组")
    return [item.strip() for item in value]


def _required_string(payload: dict[str, Any], path: Path, line: int, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line} 缺少非空字段 {field}")
    return value.strip()


def _optional_string(value: Any, path: Path, line: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{path}:{line} 的可选依据字段必须为字符串")
    return value.strip() or None
