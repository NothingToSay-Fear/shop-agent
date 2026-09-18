from datetime import date
from types import SimpleNamespace

import pytest

import app.agent.workflow as workflow_module
from app.agent.execution_plan import build_execution_plan, validate_execution_plan
from app.agent.tools.tracker import AgentToolTracker, ToolCallAudit
from app.config import Settings
from app.services.conversation_context import ConversationContextSnapshot
from app.agent.data_query_agent import DataQueryAgent
from app.agent.data_query_agent import DataQueryObservation
from app.agent.task_orchestrator import DelegationPlan
from app.services.task_plans import ExecutablePlanAction
from app.services.intent_router import RetrievalRoute


@pytest.mark.parametrize(
    ("route_mode", "expected_tools"),
    [
        ("metrics", ("query_metric_rag",)),
        ("knowledge", ("query_knowledge_rag",)),
        ("web", ("search_web",)),
        ("hybrid", ("query_metric_rag", "query_knowledge_rag")),
        (
            "web_hybrid",
            ("query_metric_rag", "query_knowledge_rag", "search_web"),
        ),
    ],
)
def test_route_is_converted_to_a_fixed_execution_plan(
    route_mode: str, expected_tools: tuple[str, ...]
) -> None:
    """路由必须由代码映射为必调工具清单，不能让模型自行删减。"""
    route = RetrievalRoute(route_mode, None, 0.9, False)  # type: ignore[arg-type]

    plan = build_execution_plan(route)

    assert plan.required_tools == expected_tools


def test_successful_tool_requires_matching_reference_id() -> None:
    """工具声明成功时必须提供所属类型的真实引用标识。"""
    plan = build_execution_plan(RetrievalRoute("metrics", None, 0.9, False))
    invalid_call = ToolCallAudit(
        tool_name="query_metric_rag",
        input_summary="问题长度：10 个字符",
        result_summary="命中 1 个经营指标",
        reference_ids=("knowledge_chunk:wrong",),
        status="success",
        duration_ms=1,
    )

    validation = validate_execution_plan(plan, [invalid_call])

    assert validation.passed is False
    assert validation.invalid_reference_tools == ("query_metric_rag",)


def test_tool_call_reservation_limits_repeated_real_execution() -> None:
    """执行计划完成后，模型再次请求同一工具不得触发第二次真实检索。"""
    tracker = AgentToolTracker()

    assert tracker.reserve_tool_call("query_metric_rag", 1) is True
    assert tracker.reserve_tool_call("query_metric_rag", 1) is False


@pytest.mark.asyncio
async def test_workflow_executes_plan_before_real_model_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实模型路径也必须先产生受控工具轨迹，再进入回答生成。"""
    events: list[str] = []

    class FakeTool:
        name = "query_metric_rag"

        async def ainvoke(self, _: dict[str, str | None]) -> str:
            events.append("query_metric_rag")
            tracker.record_tool_call(
                tool_name="query_metric_rag",
                input_summary="问题长度：8 个字符",
                result_summary="命中 1 个经营指标",
                reference_ids=("metric:paid_gmv",),
                status="success",
                started_at=tracker.start_tool_call(),
            )
            return "支付 GMV：100 元"

    def fake_build_agent_tools(
        active_tracker: AgentToolTracker,
        _: Settings,
        __: str | None = None,
        ___: object | None = None,
    ) -> list[FakeTool]:
        nonlocal tracker
        tracker = active_tracker
        return [FakeTool()]

    tracker = AgentToolTracker()
    monkeypatch.setattr(workflow_module, "build_agent_tools", fake_build_agent_tools)
    workflow = workflow_module.AgentWorkflow(Settings(llm_api_key="test", llm_model="test-model"))

    async def fake_generate_with_llm(*_: object) -> str:
        assert events == ["query_metric_rag"]
        events.append("llm")
        return "基于受控结果的回答"

    monkeypatch.setattr(workflow.answer_generator, "generate_with_llm", fake_generate_with_llm)

    result = await workflow.answer("GMV 是多少", "metrics")

    assert result.answer == "基于受控结果的回答"
    assert events == ["query_metric_rag", "llm"]
    assert result.tracker.reference_ids == ["metric:paid_gmv"]


@pytest.mark.asyncio
async def test_workflow_passes_confirmed_context_to_controlled_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """短问题进入工具前必须附带已确认条件，而不是依赖模型回忆聊天原文。"""
    captured_payload: dict[str, str | None] = {}

    class FakeTool:
        name = "query_metric_rag"

        async def ainvoke(self, payload: dict[str, str | None]) -> str:
            captured_payload.update(payload)
            tracker.record_tool_call(
                tool_name="query_metric_rag",
                input_summary="测试上下文传递",
                result_summary="命中 1 个经营指标",
                reference_ids=("metric:paid_order_count",),
                status="success",
                started_at=tracker.start_tool_call(),
            )
            return "支付订单数：100"

    def fake_build_agent_tools(
        active_tracker: AgentToolTracker,
        _: Settings,
        __: str | None = None,
        ___: object | None = None,
    ) -> list[FakeTool]:
        nonlocal tracker
        tracker = active_tracker
        return [FakeTool()]

    tracker = AgentToolTracker()
    monkeypatch.setattr(workflow_module, "build_agent_tools", fake_build_agent_tools)
    workflow = workflow_module.AgentWorkflow(Settings(llm_api_key=None, llm_model=None))
    context = ConversationContextSnapshot(
        activity="618",
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 20),
        metric_hints=("支付订单数",),
    )

    await workflow.answer(
        "再看看订单量",
        "metrics",
        conversation_context=context,
        data_query_agent=DataQueryAgent(
            {"metrics": ["paid_gmv", "conversion_rate", "refund_rate"]},
            "分析 GMV、支付转化率和退款率",
        ),
    )

    assert captured_payload["question"] is not None
    assert "已确认会话查询条件" in captured_payload["question"]
    assert "活动=618" in captured_payload["question"]
    assert "指标=支付订单数" in captured_payload["question"]
    assert captured_payload["start_date"] == date(2026, 6, 1)
    assert captured_payload["end_date"] == date(2026, 6, 20)
    assert captured_payload["metric_codes"] == ["paid_gmv", "conversion_rate", "refund_rate"]


@pytest.mark.asyncio
async def test_persisted_action_graph_replans_then_preserves_final_actions() -> None:
    """LangGraph 必须在证据不足时走重规划分支，并保留延后执行的最终动作。"""

    def action(
        key: str,
        action_type: str,
        *,
        depends_on: tuple[str, ...] = (),
        tool_name: str | None = None,
        action_input: dict[str, object] | None = None,
    ) -> ExecutablePlanAction:
        return ExecutablePlanAction(
            step_id=key,
            key=key,
            title=key,
            action_type=action_type,
            tool_name=tool_name,
            depends_on=depends_on,
            action_input=action_input or {},
            expected_output=None,
            capability_requirement={},
            condition={},
            plan_version=1,
        )

    class FakeController:
        def __init__(self) -> None:
            self.plan = SimpleNamespace(budget={"max_tool_calls": 4})
            self.finished: list[tuple[str, str]] = []
            self.replan_reasons: list[str] = []

        async def begin(self, current: ExecutablePlanAction) -> str:
            return current.key

        async def finish(
            self,
            current: ExecutablePlanAction,
            _: object,
            *,
            status: str,
            result_summary: str,
            evidence_references: tuple[str, ...] = (),
            error_message: str | None = None,
        ) -> None:
            del result_summary, evidence_references, error_message
            self.finished.append((current.key, status))

        async def append_replan(self, _: object, reason: str) -> list[ExecutablePlanAction]:
            self.replan_reasons.append(reason)
            return [
                action("confirm_constraints", "confirm_constraints"),
                action(
                    "query_metrics",
                    "query_metrics",
                    depends_on=("confirm_constraints",),
                    tool_name="query_metric_rag",
                ),
                action(
                    "followup_metrics",
                    "query_metrics",
                    depends_on=("evaluate_evidence",),
                    tool_name="query_metric_rag",
                    action_input={"metric_codes": ["paid_order_count"]},
                ),
                action(
                    "followup_evaluation",
                    "evaluate_evidence",
                    depends_on=("followup_metrics",),
                ),
            ]

    class FakeTool:
        name = "query_metric_rag"

        async def ainvoke(self, _: dict[str, object]) -> str:
            tracker.record_tool_call(
                tool_name=self.name,
                input_summary="test metric query",
                result_summary="metric result",
                reference_ids=("metric:paid_gmv",),
                status="success",
                started_at=tracker.start_tool_call(),
            )
            return "metric result"

    class FakeDataQueryAgent:
        def __init__(self) -> None:
            self.executions = 0

        async def execute(self, metric_tool: FakeTool, *_: object, **__: object) -> None:
            self.executions += 1
            await metric_tool.ainvoke({})

        def observe(self, _: AgentToolTracker) -> DataQueryObservation:
            if self.executions == 1:
                return DataQueryObservation(False, "need a driver metric", ("paid_order_count",))
            return DataQueryObservation(True, "evidence is sufficient")

    tracker = AgentToolTracker()
    controller = FakeController()
    data_agent = FakeDataQueryAgent()
    workflow = workflow_module.AgentWorkflow(Settings(llm_api_key=None, llm_model=None))

    deferred = await workflow._run_persisted_action_graph(
        [FakeTool()],
        "why did GMV decline",
        tracker,
        None,
        ConversationContextSnapshot(),
        data_agent,  # type: ignore[arg-type]
        DelegationPlan("analysis", "metrics", False, ()),
        [
            action("confirm_constraints", "confirm_constraints"),
            action(
                "query_metrics",
                "query_metrics",
                depends_on=("confirm_constraints",),
                tool_name="query_metric_rag",
            ),
            action("evaluate_evidence", "evaluate_evidence", depends_on=("query_metrics",)),
            action("synthesize", "synthesize", depends_on=("evaluate_evidence",)),
        ],
        controller,  # type: ignore[arg-type]
    )

    assert data_agent.executions == 2
    assert controller.replan_reasons == ["need a driver metric"]
    assert deferred["synthesize"].key == "synthesize"
    assert ("followup_evaluation", "completed") in controller.finished
