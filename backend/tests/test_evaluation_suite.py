"""验证固定运营场景在受控编排中的路由、工具、引用与降级行为。"""

from __future__ import annotations

import pytest

import app.agent.workflow as workflow_module
from app.agent.tools.tracker import AgentToolTracker
from app.agent.workflow import AgentWorkflow
from app.config import Settings
from app.services.intent_router import RetrievalRoute
from app.services.knowledge_rag import KnowledgeQueryContext
from app.services.metric_rag import MetricQueryContext
from app.services.web_search import WebSearchContext, WebSearchItem
from tests.evaluation_cases import EVALUATION_CASES, EvaluationCase, EvaluationToolResult


class EvaluationTool:
    """以固定资料模拟真实工具边界，不连接数据库或外部网络。"""

    def __init__(
        self, name: str, tracker: AgentToolTracker, case: EvaluationCase
    ) -> None:
        self.name = name
        self.tracker = tracker
        self.case = case

    async def ainvoke(self, payload: dict[str, str | None]) -> str:
        """将样例定义的结果写入真实 Tracker，使后续校验逻辑保持不变。"""
        assert payload["question"] == self.case.question
        result = self.case.tool_results.get(self.name)
        assert result is not None, f"{self.case.case_id} 缺少 {self.name} 的评估结果"
        started_at = self.tracker.start_tool_call()
        self._apply_context(result)
        self.tracker.record_tool_call(
            tool_name=self.name,
            input_summary="固定评估输入",
            result_summary="固定评估结果",
            reference_ids=result.reference_ids,
            status=result.status,
            started_at=started_at,
            error_code=result.error_code,
        )
        return result.context_text or "无可用资料"

    def _apply_context(self, result: EvaluationToolResult) -> None:
        """按工具类型写入与生产环境相同的最小上下文。"""
        if self.name == "query_metric_rag" and result.context_text:
            self.tracker.metric_context = MetricQueryContext(
                text=result.context_text,
                metric_codes=tuple(reference.removeprefix("metric:") for reference in result.reference_ids),
            )
        elif self.name == "query_knowledge_rag":
            if result.context_text:
                self.tracker.knowledge_context = KnowledgeQueryContext(
                    text=result.context_text,
                    references="固定评估知识库资料",
                    reference_ids=result.reference_ids,
                )
            elif result.status == "empty":
                self.tracker.knowledge_miss = True
        elif self.name == "search_web":
            if result.context_text:
                self.tracker.web_context = WebSearchContext(
                    query=self.case.question,
                    items=[
                        WebSearchItem(
                            title="平台公开公告",
                            url=result.reference_ids[0],
                            content=result.context_text,
                        )
                    ],
                )
            else:
                self.tracker.web_search_miss = True


def test_evaluation_cases_cover_all_seeded_activities() -> None:
    """固定评估集必须覆盖三份已初始化的活动测试资料。"""
    activities = {case.activity for case in EVALUATION_CASES}

    assert {"618", "七夕", "春季"}.issubset(activities)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", EVALUATION_CASES, ids=lambda case: case.case_id)
async def test_agent_workflow_matches_fixed_evaluation_case(
    monkeypatch: pytest.MonkeyPatch, case: EvaluationCase
) -> None:
    """每条样例均应执行预期工具、保留合法引用，并满足稳定的回答或降级要求。"""

    async def fake_resolve_route(
        self: AgentWorkflow, _: str, __: str
    ) -> RetrievalRoute:
        return RetrievalRoute(case.route_mode, None, 1.0, False)

    def fake_build_agent_tools(
        tracker: AgentToolTracker, _: Settings, __: str | None = None
    ) -> list[EvaluationTool]:
        return [
            EvaluationTool("query_metric_rag", tracker, case),
            EvaluationTool("query_knowledge_rag", tracker, case),
            EvaluationTool("search_web", tracker, case),
        ]

    monkeypatch.setattr(AgentWorkflow, "_resolve_route", fake_resolve_route)
    monkeypatch.setattr(workflow_module, "build_agent_tools", fake_build_agent_tools)
    workflow = AgentWorkflow(Settings(llm_api_key=None, llm_model=None))

    result = await workflow.answer(case.question, None, "hybrid")

    assert result.tracker.route is not None
    assert result.tracker.route.mode == case.route_mode
    assert tuple(call.tool_name for call in result.tracker.tool_calls) == case.expected_tools
    assert all(call.status in {"success", "empty", "skipped"} for call in result.tracker.tool_calls)
    assert all(
        len([call for call in result.tracker.tool_calls if call.tool_name == tool_name]) == 1
        for tool_name in case.expected_tools
    )
    assert _references_match_tool_types(result.tracker) is True
    for term in case.expected_answer_terms:
        assert term in result.answer


def _references_match_tool_types(tracker: AgentToolTracker) -> bool:
    """复用评估中的最小引用规则，避免只检查“引用数量不为零”。"""
    prefixes = {
        "query_metric_rag": ("metric:",),
        "query_knowledge_rag": ("knowledge_chunk:",),
        "search_web": ("https://", "http://"),
    }
    for call in tracker.tool_calls:
        if call.status != "success":
            continue
        expected_prefixes = prefixes[call.tool_name]
        if not call.reference_ids or not all(
            reference.startswith(expected_prefixes) for reference in call.reference_ids
        ):
            return False
    return True
