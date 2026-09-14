"""在隔离数据库中执行真实 RAG、Agent 与回答规则评测。"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from app.agent.workflow import AgentWorkflow
from app.config import Settings
from app.database import SessionLocal
from app.evaluation.dataset import EvaluationCase, EvaluationDataset, load_dataset
from app.evaluation.metrics import mean, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from app.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeIndexJob,
    User,
    UserKnowledgeDocumentSetting,
)
from app.seed import seed_demo_data
from app.services.document_parser import apply_semantic_boundaries, parse_document
from app.services.conversation_context import (
    ConversationContextSnapshot,
    build_context_snapshot,
    build_retrieval_question,
)
from app.services.knowledge_rag import trace_knowledge_retrieval
from app.services.knowledge_search import build_knowledge_retrieval_text, build_knowledge_search_terms
from app.services.local_embeddings import embed_texts

EVALUATION_USER_ID = "a76336a4-7a04-5a79-a9be-b575c7c03db5"
EVALUATION_USERNAME = "__rag_evaluation_v1__"

# 报告不应只输出英文键和小数；定义同时供 Markdown 与 JSON 消费者使用。
METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "route_accuracy": {
        "name": "路由准确率",
        "term": "Route Accuracy",
        "meaning": "问题被正确判断为指标查询、知识库问答或综合查询的比例。",
        "direction": "越高越好",
    },
    "plan_accuracy": {
        "name": "执行计划准确率",
        "term": "Plan Accuracy",
        "meaning": "实际调用的工具种类与顺序完全符合预期的比例。",
        "direction": "越高越好",
    },
    "metric_selection_recall": {
        "name": "指标选择命中率",
        "term": "Metric Selection Recall",
        "meaning": "指标 RAG 是否选择了问题所需指标，例如 GMV 或支付订单数。",
        "direction": "越高越好",
    },
    "knowledge_dense_recall_at_40": {
        "name": "向量候选召回率@40",
        "term": "Dense Recall@40",
        "meaning": "正确资料进入 pgvector HNSW 向量检索前 40 个候选的比例。",
        "direction": "越高越好",
    },
    "knowledge_sparse_recall_at_40": {
        "name": "全文候选召回率@40",
        "term": "Sparse Recall@40",
        "meaning": "正确资料进入 GIN 全文检索前 40 个候选的比例。",
        "direction": "越高越好",
    },
    "knowledge_fused_recall_at_40": {
        "name": "融合候选召回率@40",
        "term": "RRF Recall@40",
        "meaning": "向量与全文结果经 RRF 融合后，正确资料仍在前 40 个候选中的比例。",
        "direction": "越高越好",
    },
    "knowledge_final_recall_at_4": {
        "name": "最终召回率@4",
        "term": "Final Recall@4",
        "meaning": "经过 CrossEncoder 精排后，正确资料仍在最终 Top-4 中的比例。",
        "direction": "越高越好",
    },
    "knowledge_final_mrr": {
        "name": "首条正确依据排名",
        "term": "MRR",
        "meaning": "第一条正确资料的排名倒数；正确资料越靠前，数值越接近 1。",
        "direction": "越高越好",
    },
    "knowledge_final_ndcg_at_4": {
        "name": "最终排序质量@4",
        "term": "nDCG@4",
        "meaning": "同时有多条正确资料时，衡量它们在 Top-4 中是否整体靠前。",
        "direction": "越高越好",
    },
    "knowledge_final_precision_at_4": {
        "name": "最终结果精确率@4",
        "term": "Precision@4",
        "meaning": "最终 Top-4 中真正相关资料的比例，用于观察上下文噪声。",
        "direction": "越高越好",
    },
    "knowledge_empty_accuracy": {
        "name": "无答案识别准确率",
        "term": "No-answer Accuracy",
        "meaning": "资料不足时，系统能正确返回“未检索到可靠资料”而非编造答案的比例。",
        "direction": "越高越好",
    },
    "answer_claim_coverage": {
        "name": "关键事实覆盖率",
        "term": "Claim Coverage",
        "meaning": "回答中覆盖人工标注关键结论的比例，不要求自然语言逐字一致。",
        "direction": "越高越好",
    },
    "answer_forbidden_claim_free": {
        "name": "禁止事实规避率",
        "term": "Forbidden-claim Free Rate",
        "meaning": "回答没有出现人工标注错误事实的比例，用于发现明显编造。",
        "direction": "越高越好",
    },
    "latency_p95_ms": {
        "name": "端到端 P95 时延",
        "term": "P95 Latency",
        "meaning": "95% 的评测请求在该毫秒数内完成，包含检索、精排和回答整理。",
        "direction": "越低越好",
    },
}


@dataclass(frozen=True)
class ChunkMetadata:
    """评测中用于核对人工锚点的最小 Chunk 元数据。"""

    document: str
    heading_path: str | None
    content: str


@dataclass(frozen=True)
class CaseResult:
    """单条样例的可追溯结果，不记录任何真实用户问题或回答。"""

    case_id: str
    category: str
    route: str
    expected_route: str
    tools: tuple[str, ...]
    expected_tools: tuple[str, ...]
    metric_codes: tuple[str, ...]
    expected_metric_codes: tuple[str, ...]
    dense_documents: tuple[str, ...]
    sparse_documents: tuple[str, ...]
    fused_documents: tuple[str, ...]
    final_documents: tuple[str, ...]
    expected_documents: tuple[str, ...]
    route_ok: bool
    plan_ok: bool
    metric_ok: bool
    evidence_ok: bool
    empty_ok: bool
    answer_claim_coverage: float
    forbidden_claim_hit: bool
    duration_ms: int


async def prepare_evaluation_corpus(dataset: EvaluationDataset, settings: Settings) -> None:
    """以固定文件建立独立评测用户资料；只删除此前同一评测身份的资料。"""
    if not settings.local_embedding_enabled:
        raise RuntimeError("真实 RAG 评测需要配置 LOCAL_EMBEDDING_MODEL_PATH")
    await seed_demo_data()
    async with SessionLocal() as session:
        await _clear_previous_evaluation_documents(session)
        user = await session.get(User, EVALUATION_USER_ID)
        if user is None:
            session.add(
                User(
                    id=EVALUATION_USER_ID,
                    username=EVALUATION_USERNAME,
                    display_name="RAG 评测专用用户",
                    password_hash="evaluation-account-disabled",
                    is_admin=False,
                )
            )
            await session.flush()

        for filename in dataset.documents:
            source_path = dataset.root / "documents" / filename
            if not source_path.is_file():
                raise ValueError(f"评测语料文件不存在：{source_path}")
            raw_content = source_path.read_bytes()
            parsed = parse_document(filename, raw_content)
            parsed = await apply_semantic_boundaries(
                parsed,
                settings.knowledge_chunk_semantic_similarity_threshold,
                lambda texts: embed_texts(texts, settings),
                settings.knowledge_index_batch_size,
            )
            embeddings = await embed_texts(
                [
                    build_knowledge_retrieval_text(
                        source_path.stem, chunk.heading_path, chunk.content_type, chunk.content
                    )
                    for chunk in parsed.chunks
                ],
                settings,
            )
            if embeddings is None:
                raise RuntimeError("本地嵌入模型不可用，无法建立真实评测索引")
            document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"shop-agent-evaluation:{filename}"))
            document = KnowledgeDocument(
                id=document_id,
                owner_user_id=EVALUATION_USER_ID,
                space="private",
                title=source_path.stem,
                original_filename=filename,
                file_type=source_path.suffix.removeprefix(".").lower() or "txt",
                file_path=str(source_path.resolve()),
                content=parsed.content,
                status="ready",
                chunk_count=len(parsed.chunks),
            )
            session.add(document)
            # 当前评测装载器按显式 ID 写入，模型未声明 ORM relationship；先落库父记录以满足 Chunk 外键。
            await session.flush()
            session.add(
                UserKnowledgeDocumentSetting(
                    user_id=EVALUATION_USER_ID,
                    document_id=document_id,
                    retrieval_enabled=True,
                )
            )
            session.add_all(
                KnowledgeChunk(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{index}")),
                    document_id=document_id,
                    chunk_index=index,
                    content=chunk.content,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    heading=chunk.heading,
                    heading_path=chunk.heading_path,
                    content_type=chunk.content_type,
                    embedding_vector=embedding,
                    search_terms=build_knowledge_search_terms(
                        document.title, chunk.heading_path, chunk.content_type, chunk.content
                    ),
                    embedding_model=settings.local_embedding_model_id,
                )
                for index, (chunk, embedding) in enumerate(zip(parsed.chunks, embeddings, strict=True))
            )
        await session.commit()


async def run_evaluation(dataset: EvaluationDataset, settings: Settings) -> dict[str, Any]:
    """执行真实检索与工作流，聚合质量指标和质量门禁结果。"""
    results: list[CaseResult] = []
    async with SessionLocal() as session:
        for case in dataset.cases:
            results.append(await _evaluate_case(session, case, settings))
    metrics = _aggregate_metrics(results)
    gate_failures = _quality_gate_failures(metrics, dataset.quality_gates)
    baseline_failures = _baseline_regression_failures(dataset.root / "baseline.json", metrics)
    return {
        "dataset_version": dataset.version,
        "generated_at": datetime.now(UTC).isoformat(),
        "models": {
            "embedding": settings.local_embedding_model_id,
            "reranker": settings.local_reranker_model_id if settings.local_reranker_enabled else None,
        },
        "metrics": metrics,
        "metric_definitions": METRIC_DEFINITIONS,
        "quality_gates": dataset.quality_gates,
        "gate_failures": gate_failures,
        "baseline_failures": baseline_failures,
        "passed": not gate_failures and not baseline_failures,
        "cases": [asdict(result) for result in results],
    }


async def _evaluate_case(session, case: EvaluationCase, settings: Settings) -> CaseResult:
    """同一问题同时记录真实工作流终态和知识库检索各阶段排名。"""
    started_at = time.perf_counter()
    workflow = AgentWorkflow(settings)
    context = _build_evaluation_context(case.context_turns)
    retrieval_question = build_retrieval_question(case.question, context)
    workflow_result = await workflow.answer(
        case.question,
        "hybrid",
        user_id=EVALUATION_USER_ID,
        conversation_context=context,
    )
    route = workflow_result.tracker.route.mode if workflow_result.tracker.route else "unknown"
    tools = tuple(call.tool_name for call in workflow_result.tracker.tool_calls)
    metric_codes = workflow_result.tracker.metric_context.metric_codes if workflow_result.tracker.metric_context else ()

    dense_documents: tuple[str, ...] = ()
    sparse_documents: tuple[str, ...] = ()
    fused_documents: tuple[str, ...] = ()
    final_documents: tuple[str, ...] = ()
    final_chunk_ids: tuple[str, ...] = ()
    chunk_metadata: dict[str, ChunkMetadata] = {}
    if "query_knowledge_rag" in case.expected_tools:
        trace = await trace_knowledge_retrieval(session, EVALUATION_USER_ID, retrieval_question, settings)
        chunk_metadata = await _chunk_metadata_for_ids(
            session,
            {
                chunk_id
                for ranking in (*trace.dense_rankings, *trace.sparse_rankings)
                for chunk_id in ranking
            }
            | set(trace.fused_ranking)
            | set(trace.final_ranking),
        )
        dense_documents = _unique_documents(
            chunk_metadata, (chunk_id for ranking in trace.dense_rankings for chunk_id in ranking)
        )
        sparse_documents = _unique_documents(
            chunk_metadata, (chunk_id for ranking in trace.sparse_rankings for chunk_id in ranking)
        )
        fused_documents = _unique_documents(chunk_metadata, trace.fused_ranking)
        final_documents = _unique_documents(chunk_metadata, trace.final_ranking)
        final_chunk_ids = trace.final_ranking

    expected_documents = tuple(target.document for target in case.evidence)
    evidence_ok = _evidence_targets_match(case, final_chunk_ids, chunk_metadata)
    knowledge_empty = not final_documents
    claims = tuple(claim for claim in case.expected_claims if claim in workflow_result.answer)
    claim_coverage = len(claims) / len(case.expected_claims) if case.expected_claims else 1.0
    forbidden_hit = any(claim in workflow_result.answer for claim in case.forbidden_claims)
    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        route=route,
        expected_route=case.expected_route,
        tools=tools,
        expected_tools=case.expected_tools,
        metric_codes=tuple(metric_codes),
        expected_metric_codes=case.expected_metric_codes,
        dense_documents=dense_documents,
        sparse_documents=sparse_documents,
        fused_documents=fused_documents,
        final_documents=final_documents,
        expected_documents=expected_documents,
        route_ok=route == case.expected_route,
        plan_ok=tools == case.expected_tools,
        metric_ok=set(case.expected_metric_codes).issubset(metric_codes),
        evidence_ok=evidence_ok,
        empty_ok=knowledge_empty == case.expected_knowledge_empty if case.expected_knowledge_empty else True,
        answer_claim_coverage=claim_coverage,
        forbidden_claim_hit=forbidden_hit,
        duration_ms=round((time.perf_counter() - started_at) * 1000),
    )


async def _clear_previous_evaluation_documents(session) -> None:
    document_ids = select(KnowledgeDocument.id).where(
        KnowledgeDocument.owner_user_id == EVALUATION_USER_ID
    )
    await session.execute(
        delete(UserKnowledgeDocumentSetting).where(
            UserKnowledgeDocumentSetting.document_id.in_(document_ids)
        )
    )
    await session.execute(delete(KnowledgeIndexJob).where(KnowledgeIndexJob.document_id.in_(document_ids)))
    await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.document_id.in_(document_ids)))
    await session.execute(delete(KnowledgeDocument).where(KnowledgeDocument.id.in_(document_ids)))
    await session.commit()


async def _chunk_metadata_for_ids(session, chunk_ids: set[str]) -> dict[str, ChunkMetadata]:
    if not chunk_ids:
        return {}
    result = await session.execute(
        select(
            KnowledgeChunk.id,
            KnowledgeDocument.original_filename,
            KnowledgeChunk.heading_path,
            KnowledgeChunk.content,
        )
        .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
        .where(KnowledgeChunk.id.in_(chunk_ids))
    )
    return {
        chunk_id: ChunkMetadata(document, heading_path, content)
        for chunk_id, document, heading_path, content in result.all()
    }


def _unique_documents(chunk_metadata: dict[str, ChunkMetadata], chunk_ids) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            chunk_metadata[chunk_id].document for chunk_id in chunk_ids if chunk_id in chunk_metadata
        )
    )


def _evidence_targets_match(
    case: EvaluationCase, final_chunk_ids: tuple[str, ...], chunk_metadata: dict[str, ChunkMetadata]
) -> bool:
    """每条人工金标均须在最终片段中命中指定来源及可选标题、正文锚点。"""
    if not case.evidence:
        return True
    selected = [chunk_metadata[item] for item in final_chunk_ids if item in chunk_metadata]
    return all(
        any(
            chunk.document == target.document
            and (target.heading_contains is None or target.heading_contains in (chunk.heading_path or ""))
            and (target.anchor_text is None or target.anchor_text in chunk.content)
            for chunk in selected
        )
        for target in case.evidence
    )


def _build_evaluation_context(turns: tuple[str, ...]) -> ConversationContextSnapshot:
    """以生产相同的确定性规则重放前序用户问题，不持久化任何评测会话。"""
    snapshot = ConversationContextSnapshot()
    for index, question in enumerate(turns, start=1):
        snapshot = build_context_snapshot(snapshot, question, f"evaluation-turn-{index}").snapshot
    return snapshot


def _aggregate_metrics(results: list[CaseResult]) -> dict[str, float]:
    """把路由、工具、指标、资料、回答和时延汇总为稳定的 CI 指标。"""
    knowledge = [item for item in results if item.expected_documents]
    metric_cases = [item for item in results if item.expected_metric_codes]
    empty_cases = [item for item in results if item.category == "knowledge_empty"]
    return {
        "route_accuracy": mean(item.route_ok for item in results),
        "plan_accuracy": mean(item.plan_ok for item in results),
        "metric_selection_recall": mean(item.metric_ok for item in metric_cases),
        "knowledge_dense_recall_at_40": mean(
            recall_at_k(item.dense_documents, item.expected_documents, 40) for item in knowledge
        ),
        "knowledge_sparse_recall_at_40": mean(
            recall_at_k(item.sparse_documents, item.expected_documents, 40) for item in knowledge
        ),
        "knowledge_fused_recall_at_40": mean(
            recall_at_k(item.fused_documents, item.expected_documents, 40) for item in knowledge
        ),
        "knowledge_final_recall_at_4": mean(
            recall_at_k(item.final_documents, item.expected_documents, 4) for item in knowledge
        ),
        "knowledge_final_mrr": mean(
            reciprocal_rank(item.final_documents, item.expected_documents) for item in knowledge
        ),
        "knowledge_final_ndcg_at_4": mean(
            ndcg_at_k(item.final_documents, item.expected_documents, 4) for item in knowledge
        ),
        "knowledge_final_precision_at_4": mean(
            precision_at_k(item.final_documents, item.expected_documents, 4) for item in knowledge
        ),
        "knowledge_empty_accuracy": mean(item.empty_ok for item in empty_cases),
        "answer_claim_coverage": mean(item.answer_claim_coverage for item in results),
        "answer_forbidden_claim_free": mean(not item.forbidden_claim_hit for item in results),
        "latency_p95_ms": _percentile([item.duration_ms for item in results], 0.95),
    }


def _percentile(values: list[int], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * ratio))
    return float(ordered[index])


def _quality_gate_failures(metrics: dict[str, float], gates: dict[str, float]) -> list[str]:
    return [
        f"{name}={metrics.get(name, 0.0):.4f} 低于最低门槛 {minimum:.4f}"
        for name, minimum in gates.items()
        if name in metrics and metrics[name] < minimum
    ]


def _baseline_regression_failures(baseline_path: Path, metrics: dict[str, float]) -> list[str]:
    """仅与人工确认过的基线比较，避免首次运行被不存在的历史值阻断。"""
    if not baseline_path.exists():
        return []
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_metrics = baseline.get("metrics", {})
    tolerance = float(baseline.get("max_regression", 0.03))
    return [
        f"{name}={value:.4f} 相比基线 {float(baseline_metrics[name]):.4f} 下降超过 {tolerance:.4f}"
        for name, value in metrics.items()
        if name in baseline_metrics and value < float(baseline_metrics[name]) - tolerance
    ]


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """输出机器可读 JSON 和便于人工验收的 Markdown 报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"rag-evaluation-{stamp}.json"
    markdown_path = output_dir / f"rag-evaluation-{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def write_baseline(report: dict[str, Any], dataset_root: Path, max_regression: float = 0.03) -> Path:
    """由人工确认后的运行结果生成下一次 CI 使用的对比基线。"""
    target = dataset_root / "baseline.json"
    target.write_text(
        json.dumps(
            {
                "dataset_version": report["dataset_version"],
                "generated_at": report["generated_at"],
                "max_regression": max_regression,
                "metrics": report["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def _render_markdown(report: dict[str, Any]) -> str:
    metric_rows = "\n".join(
        (
            f"| `{name}` | {report['metric_definitions'].get(name, {}).get('name', name)} "
            f"({report['metric_definitions'].get(name, {}).get('term', '-')}) | {value:.4f} | "
            f"{report['metric_definitions'].get(name, {}).get('direction', '-')} | "
            f"{report['metric_definitions'].get(name, {}).get('meaning', '未定义说明')} |"
        )
        for name, value in report["metrics"].items()
    )
    failed_cases = [
        item for item in report["cases"]
        if not (item["route_ok"] and item["plan_ok"] and item["metric_ok"] and item["evidence_ok"] and item["empty_ok"])
    ]
    failure_rows = "\n".join(
        f"| {item['case_id']} | 路由={item['route_ok']}；计划={item['plan_ok']}；指标={item['metric_ok']}；依据={item['evidence_ok']} |"
        for item in failed_cases
    ) or "| 无 | 全部通过结构化校验 |"
    failures = "\n".join(f"- {item}" for item in [*report["gate_failures"], *report["baseline_failures"]]) or "- 无"
    return f"""# RAG 评测报告

- 数据集：`{report['dataset_version']}`
- 生成时间：{report['generated_at']}
- 嵌入模型：`{report['models']['embedding']}`
- 精排模型：`{report['models']['reranker'] or '未启用，使用 RRF 降级顺序'}`
- 结果：{'通过' if report['passed'] else '未通过'}

## 汇总指标

| 指标键 | 中文名称 / 术语 | 数值 | 判断方向 | 这个指标在测什么 |
| --- | --- | ---: | --- | --- |
{metric_rows}

## 门禁与基线失败项

{failures}

## 未通过的结构化样例

| 样例 | 未通过项 |
| --- | --- |
{failure_rows}
"""
