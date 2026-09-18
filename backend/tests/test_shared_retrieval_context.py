import pytest

import app.agent.workflow as workflow_module
from app.agent.tools.tracker import AgentToolTracker
from app.agent.workflow import AgentWorkflow
from app.config import Settings
from app.services.conversations.context import ConversationContextSnapshot
from app.services.intent_router import RetrievalRoute
from app.services.retrieval.query_expansion import PreparedRetrievalQueries


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


@pytest.mark.asyncio
async def test_route_uses_raw_question_but_retrieval_uses_confirmed_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧会话条件只能限定检索范围，不能把本轮的单一意图改判为综合检索。"""
    routed_questions: list[str] = []
    prepared_questions: list[tuple[str, list[float] | None]] = []

    async def fake_resolve_route(
        _: AgentWorkflow, question: str, __: str
    ) -> RetrievalRoute:
        routed_questions.append(question)
        return RetrievalRoute("metrics", [0.1, 0.2], 0.9, False)

    async def fake_prepare(
        question: str, _: Settings, original_embedding: list[float] | None
    ) -> PreparedRetrievalQueries:
        prepared_questions.append((question, original_embedding))
        return PreparedRetrievalQueries(question, (question,), ([0.3, 0.4],))

    class FakeTool:
        name = "query_metric_rag"

        def __init__(self, tracker: AgentToolTracker) -> None:
            self.tracker = tracker

        async def ainvoke(self, _: dict[str, object]) -> str:
            self.tracker.record_tool_call(
                tool_name=self.name,
                input_summary="上下文范围测试",
                result_summary="完成",
                reference_ids=("metric:paid_order_count",),
                status="success",
                started_at=self.tracker.start_tool_call(),
            )
            return "完成"

    def fake_build_tools(
        tracker: AgentToolTracker,
        _: Settings,
        __: str | None = None,
        ___: PreparedRetrievalQueries | None = None,
    ) -> list[FakeTool]:
        return [FakeTool(tracker)]

    monkeypatch.setattr(AgentWorkflow, "_resolve_route", fake_resolve_route)
    monkeypatch.setattr(workflow_module, "prepare_retrieval_queries", fake_prepare)
    monkeypatch.setattr(workflow_module, "build_agent_tools", fake_build_tools)

    workflow = AgentWorkflow(Settings(llm_api_key=None, llm_model=None))
    await workflow.answer(
        "再看看订单量",
        "hybrid",
        conversation_context=ConversationContextSnapshot(activity="618大促"),
    )

    assert routed_questions == ["再看看订单量"]
    assert prepared_questions == [
        ("再看看订单量\n\n已确认会话查询条件（仅用于限定本轮检索范围）：活动=618大促", None)
    ]
