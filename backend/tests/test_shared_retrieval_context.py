import pytest

import app.agent.workflow as workflow_module
from app.agent.tools.tracker import AgentToolTracker
from app.agent.workflow import AgentWorkflow
from app.config import Settings
from app.services.intent_router import RetrievalRoute
from app.services.query_expansion import PreparedRetrievalQueries


@pytest.mark.asyncio
async def test_hybrid_workflow_prepares_queries_once_and_shares_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """综合检索仅构建一次共享上下文，并注入指标与知识库工具。"""
    prepared = PreparedRetrievalQueries(
        question="结合资料分析 618 GMV 下滑原因",
        queries=("结合资料分析 618 GMV 下滑原因", "618 活动成交额下降原因"),
        query_embeddings=([0.1, 0.2], [0.3, 0.4]),
    )
    prepare_calls: list[tuple[str, list[float] | None]] = []
    injected_contexts: list[PreparedRetrievalQueries | None] = []

    async def fake_resolve_route(
        _: AgentWorkflow, __: str, ___: str
    ) -> RetrievalRoute:
        return RetrievalRoute("hybrid", [0.1, 0.2], 0.9, False)

    async def fake_prepare(
        question: str, _: Settings, original_embedding: list[float] | None
    ) -> PreparedRetrievalQueries:
        prepare_calls.append((question, original_embedding))
        return prepared

    class FakeTool:
        def __init__(self, name: str, reference_id: str, tracker: AgentToolTracker) -> None:
            self.name = name
            self.reference_id = reference_id
            self.tracker = tracker

        async def ainvoke(self, _: dict[str, object]) -> str:
            self.tracker.record_tool_call(
                tool_name=self.name,
                input_summary="共享检索上下文测试",
                result_summary="完成",
                reference_ids=(self.reference_id,),
                status="success",
                started_at=self.tracker.start_tool_call(),
            )
            return "完成"

    def fake_build_tools(
        tracker: AgentToolTracker,
        _: Settings,
        __: str | None = None,
        prepared_queries: PreparedRetrievalQueries | None = None,
    ) -> list[FakeTool]:
        injected_contexts.append(prepared_queries)
        return [
            FakeTool("query_metric_rag", "metric:paid_gmv", tracker),
            FakeTool("query_knowledge_rag", "knowledge_chunk:chunk-1", tracker),
        ]

    monkeypatch.setattr(AgentWorkflow, "_resolve_route", fake_resolve_route)
    monkeypatch.setattr(workflow_module, "prepare_retrieval_queries", fake_prepare)
    monkeypatch.setattr(workflow_module, "build_agent_tools", fake_build_tools)

    workflow = AgentWorkflow(Settings(llm_api_key=None, llm_model=None))
    result = await workflow.answer(prepared.question, "hybrid")

    assert prepare_calls == [(prepared.question, [0.1, 0.2])]
    assert injected_contexts == [prepared]
    assert tuple(call.tool_name for call in result.tracker.tool_calls) == (
        "query_metric_rag",
        "query_knowledge_rag",
    )
