from datetime import date
import pytest

from app.models import ConversationTask
from app.services.conversations.tasks import (
    TASK_COMPLETED,
    TASK_HYBRID_ANALYSIS,
    TASK_KNOWLEDGE_QA,
    TASK_METRIC_COMPARISON,
    TASK_READY,
    TASK_REVIEW,
    TASK_WAITING_CLARIFICATION,
    _continue_completed_task,
    cancel_conversation_tasks,
    prepare_conversation_task,
    task_constraint_audit,
)
from app.services.analytics.metric_rag import build_metric_query_plan
from app.services.analytics.activity_periods import resolve_activity_periods
from app.services.conversations.task_interpreter import TaskRelationshipDecision, _fallback_decision, _parse_decision
from app.services.conversations.temporal_interpreter import TemporalResolution


class _TaskSession:
    """最小会话替身：任务状态机测试不依赖数据库或模型服务。"""

    def __init__(self, scalar_results: list[object | None]) -> None:
        self.scalar_results = iter(scalar_results)
        self.added: list[ConversationTask] = []

    async def scalar(self, _statement: object) -> object | None:
        return next(self.scalar_results)

    def add(self, task: ConversationTask) -> None:
        self.added.append(task)

    async def flush(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _controlled_temporal_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """任务状态机只验证时间服务输出的消费方式，不在单元测试中调用真实模型。"""

    async def _resolve(_session: object, question: str, **_kwargs: object) -> TemporalResolution:
        lowered = question.lower()
        if "上周" in lowered and any(marker in lowered for marker in ("本周", "这周")):
            return TemporalResolution(
                "clarify", clarification="请确认本周起止日期。", source="llm"
            )
        if any(marker in lowered for marker in ("上月", "上个月")) and any(
            marker in lowered for marker in ("本月", "这个月")
        ):
            periods = (
                (("上月同期", date(2026, 8, 1), date(2026, 8, 15)), ("本月累计", date(2026, 9, 1), date(2026, 9, 15)))
                if "同期" in lowered
                else (("上月整月", date(2026, 8, 1), date(2026, 8, 31)), ("本月累计", date(2026, 9, 1), date(2026, 9, 15)))
            )
            return TemporalResolution("resolved", periods, source="llm")
        if any(marker in lowered for marker in ("上季度", "上个季度")) and any(
            marker in lowered for marker in ("本季度", "这个季度")
        ):
            return TemporalResolution(
                "resolved",
                (("上季度同期", date(2026, 4, 1), date(2026, 6, 16)), ("本季度累计", date(2026, 7, 1), date(2026, 9, 15))),
                source="llm",
            )
        if "2026-09-09" in question:
            return TemporalResolution(
                "resolved",
                (("上周", date(2026, 9, 2), date(2026, 9, 8)), ("本周", date(2026, 9, 9), date(2026, 9, 15))),
                source="llm",
            )
        return TemporalResolution("no_time")

    monkeypatch.setattr("app.services.conversations.tasks.resolve_temporal_intent", _resolve)

    async def _driver_graph(_session: object) -> dict[str, tuple[str, ...]]:
        return {
            "paid_gmv": (
                "paid_order_count",
                "visitor_count",
                "conversion_rate",
                "average_order_value",
            )
        }

    monkeypatch.setattr("app.services.conversations.tasks.load_metric_analysis_driver_graph", _driver_graph)


@pytest.mark.asyncio
async def test_metric_comparison_waits_for_range_without_querying_partial_data() -> None:
    """本周与上周未给出边界时，任务应追问，而不是先查半段数据。"""
    session = _TaskSession([None, None])

    turn = await prepare_conversation_task(
        session, "conversation-1", "message-1", "对比本周的 GMV 和上周的 GMV"
    )

    assert turn.requires_clarification is True
    assert turn.task.task_type == TASK_METRIC_COMPARISON
    assert turn.task.status == TASK_WAITING_CLARIFICATION
    assert turn.task.missing_slots == ["时间范围或数据类型"]


@pytest.mark.asyncio
async def test_causal_metric_comparison_is_an_analysis_task_with_driver_metrics() -> None:
    session = _TaskSession([None, None])

    turn = await prepare_conversation_task(
        session, "conversation-1", "message-1", "为什么本周 GMV 比上周低"
    )

    assert turn.task.task_type == TASK_HYBRID_ANALYSIS
    assert turn.task.effective_constraints["analysis_goal"] == "经营归因"
    assert turn.task.effective_constraints["data_query_plan"]["metric_codes"] == [
        "paid_gmv",
        "paid_order_count",
        "visitor_count",
        "conversion_rate",
        "average_order_value",
    ]


@pytest.mark.asyncio
async def test_task_constraints_are_the_only_execution_and_audit_source() -> None:
    session = _TaskSession([None, None])

    turn = await prepare_conversation_task(
        session,
        "conversation-1",
        "message-1",
        "为什么本周 GMV 比上周低",
    )
    audit = task_constraint_audit(turn.task)

    assert "已确认的受控指标：paid_gmv" in turn.effective_question
    assert "已确认的分析目标：经营归因" in turn.effective_question
    assert audit.snapshot == {
        "task_id": turn.task.id,
        "task_revision": 1,
        "effective_constraints": turn.task.effective_constraints,
    }
    assert audit.actions == ("本轮检索及生成仅使用会话任务的有效约束",)


@pytest.mark.asyncio
async def test_reset_cancels_completed_tasks_that_can_be_inherited() -> None:
    completed = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_METRIC_COMPARISON,
        status=TASK_COMPLETED,
        task_frame={"base_question": "对比 GMV", "supplements": []},
        effective_constraints={"metrics": ["paid_gmv"]},
        missing_slots=[],
        source_message_ids=["message-1"],
    )
    cancelled = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_METRIC_COMPARISON,
        status="cancelled",
        task_frame={"base_question": "旧任务", "supplements": []},
        missing_slots=[],
        source_message_ids=["message-0"],
    )

    class _ResetSession:
        def __init__(self) -> None:
            self.added: list[object] = []

        async def scalars(self, _statement: object) -> list[ConversationTask]:
            return [completed, cancelled]

        def add(self, value: object) -> None:
            self.added.append(value)

    await cancel_conversation_tasks(_ResetSession(), "conversation-1")  # type: ignore[arg-type]

    assert completed.status == "cancelled"
    assert cancelled.status == "cancelled"


@pytest.mark.asyncio
async def test_causal_follow_up_upgrades_a_completed_metric_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_METRIC_COMPARISON,
        status=TASK_COMPLETED,
        route_mode="metrics",
        revision=1,
        task_frame={"base_question": "本周 GMV 比上周低", "supplements": []},
        effective_constraints={"metrics": ["paid_gmv"]},
        missing_slots=[],
        source_message_ids=["message-1"],
    )

    async def _continue(*_args: object, **_kwargs: object) -> TaskRelationshipDecision:
        return TaskRelationshipDecision("continue", (), 0.95, "延续同一指标范围", "llm")

    async def _no_time(*_args: object, **_kwargs: object) -> TemporalResolution:
        return TemporalResolution("no_time", source="local_gate")

    monkeypatch.setattr("app.services.conversations.tasks.interpret_task_relationship", _continue)
    monkeypatch.setattr("app.services.conversations.tasks.resolve_temporal_intent", _no_time)
    session = _TaskSession([None, task])

    turn = await prepare_conversation_task(
        session, "conversation-1", "message-2", "为什么会低"
    )

    assert turn.task.task_type == TASK_HYBRID_ANALYSIS
    assert turn.task.route_mode == "hybrid"
    assert turn.task.effective_constraints["data_query_plan"]["metric_codes"] == [
        "paid_gmv",
        "paid_order_count",
        "visitor_count",
        "conversion_rate",
        "average_order_value",
    ]


@pytest.mark.asyncio
async def test_month_comparison_supersedes_waiting_week_task_and_uses_controlled_periods() -> None:
    """普通“上月和本月”按自然月语义查询上月整月与本月累计，不能被旧周对比劫持。"""
    pending_week_task = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_METRIC_COMPARISON,
        status=TASK_WAITING_CLARIFICATION,
        route_mode="metrics",
        task_frame={"base_question": "比较上周和本周的 GMV", "supplements": []},
        missing_slots=["本周的起止日期"],
        source_message_ids=["message-1"],
    )
    session = _TaskSession([pending_week_task, None, date(2026, 9, 15)])

    turn = await prepare_conversation_task(
        session, "conversation-1", "message-2", "比较一下上个月和这个月的 GMV"
    )
    plan = build_metric_query_plan(turn.effective_question)

    assert pending_week_task.status == "superseded"
    assert turn.requires_clarification is False
    assert "本周的起止日期" not in turn.effective_question
    assert [(unit.start_date, unit.end_date) for unit in plan.units] == [
        (date(2026, 8, 1), date(2026, 8, 31)),
        (date(2026, 9, 1), date(2026, 9, 15)),
    ]


@pytest.mark.asyncio
async def test_completed_legacy_month_task_without_periods_is_rebuilt_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧版本错误完成的月度任务重试时，不能继续复用遗留的周区间。"""
    legacy_task = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_METRIC_COMPARISON,
        status=TASK_COMPLETED,
        route_mode="metrics",
        task_frame={"base_question": "比较一下上个月和这个月的 GMV", "supplements": []},
        missing_slots=[],
        source_message_ids=["message-1"],
    )

    async def _revise(*_args: object, **_kwargs: object) -> TaskRelationshipDecision:
        return TaskRelationshipDecision("revise", ("time_range",), 0.9, "重述原月度任务", "llm")

    monkeypatch.setattr("app.services.conversations.tasks.interpret_task_relationship", _revise)
    session = _TaskSession([None, legacy_task, date(2026, 9, 15)])

    turn = await prepare_conversation_task(
        session, "conversation-1", "message-2", "比较一下上个月和这个月的 GMV"
    )
    plan = build_metric_query_plan(turn.effective_question)

    assert legacy_task.status == TASK_READY
    assert legacy_task.revision == 2
    assert [(unit.start_date, unit.end_date) for unit in plan.units] == [
        (date(2026, 8, 1), date(2026, 8, 31)),
        (date(2026, 9, 1), date(2026, 9, 15)),
    ]


@pytest.mark.asyncio
async def test_month_same_period_is_used_only_when_user_explicitly_requests_it() -> None:
    """“上月同期”才使用等天数区间，避免覆盖普通“上月”的整月含义。"""
    session = _TaskSession([None, None, date(2026, 9, 15)])

    turn = await prepare_conversation_task(
        session, "conversation-1", "message-1", "比较本月累计和上月同期的 GMV"
    )
    plan = build_metric_query_plan(turn.effective_question)

    assert [(unit.start_date, unit.end_date) for unit in plan.units] == [
        (date(2026, 8, 1), date(2026, 8, 15)),
        (date(2026, 9, 1), date(2026, 9, 15)),
    ]


@pytest.mark.asyncio
async def test_semantic_task_interpretation_can_replace_waiting_task_without_keyword_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型判断为新任务时，即使新说法没有指标关键词，也不得困在旧任务的日期追问中。"""
    pending_week_task = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_METRIC_COMPARISON,
        status=TASK_WAITING_CLARIFICATION,
        route_mode="metrics",
        task_frame={"base_question": "比较上周和本周的 GMV", "supplements": []},
        missing_slots=["本周的起止日期"],
        source_message_ids=["message-1"],
    )

    async def _replace(*_args: object, **_kwargs: object) -> TaskRelationshipDecision:
        return TaskRelationshipDecision(
            "replace", ("time_range", "analysis_goal"), 0.95, "改为独立的月度经营任务", "llm"
        )

    monkeypatch.setattr("app.services.conversations.tasks.interpret_task_relationship", _replace)
    session = _TaskSession([pending_week_task, None])

    turn = await prepare_conversation_task(
        session, "conversation-1", "message-2", "换成最近一个月的经营表现"
    )

    assert pending_week_task.status == "superseded"
    assert turn.requires_clarification is False
    assert pending_week_task.task_frame["last_interpretation"]["relation"] == "replace"


@pytest.mark.asyncio
async def test_quarter_comparison_replaces_waiting_week_task_and_uses_controlled_periods() -> None:
    """季度比较是新任务，并按业务最新日展开为本季度累计与上季度同期。"""
    pending_week_task = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_METRIC_COMPARISON,
        status=TASK_WAITING_CLARIFICATION,
        route_mode="metrics",
        task_frame={"base_question": "比较上周和本周的 GMV", "supplements": []},
        missing_slots=["本周的起止日期"],
        source_message_ids=["message-1"],
    )
    session = _TaskSession([pending_week_task, None, date(2026, 9, 15)])

    turn = await prepare_conversation_task(
        session, "conversation-1", "message-2", "比较上季度和本季度的 GMV"
    )
    plan = build_metric_query_plan(turn.effective_question)

    assert pending_week_task.status == "superseded"
    assert turn.requires_clarification is False
    assert [(unit.start_date, unit.end_date) for unit in plan.units] == [
        # 两个季度起始月份天数不同，按“季度至今”的等天数口径取上季度同期。
        (date(2026, 4, 1), date(2026, 6, 16)),
        (date(2026, 7, 1), date(2026, 9, 15)),
    ]


def test_task_interpreter_accepts_only_high_confidence_constrained_json() -> None:
    """模型只能提供限定的关系判断；低置信度或额外槽位不会影响后端任务执行。"""
    decision = _parse_decision(
        '{"relation":"replace","changed_slots":["time_range","unknown"],'
        '"confidence":0.91,"reason":"比较周期从周切换为月",'
        '"next_task_input":"比较上个月和这个月的 GMV"}'
    )

    assert decision is not None
    assert decision.relation == "replace"
    assert decision.changed_slots == ("time_range",)
    assert decision.source == "llm"
    assert decision.next_task_input == "比较上个月和这个月的 GMV"
    assert _parse_decision(
        '{"relation":"replace","changed_slots":[],"confidence":0.2,"reason":"不确定"}'
    ) is None


def test_fallback_does_not_reopen_completed_task_without_continuation_signal() -> None:
    """未配置 LLM 时，无关输入不能意外续接上一项已经完成的任务。"""
    completed_task = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_KNOWLEDGE_QA,
        status=TASK_COMPLETED,
        route_mode="knowledge",
        task_frame={"base_question": "618 活动资料中的发货规则是什么", "supplements": []},
        missing_slots=[],
        source_message_ids=["message-1"],
    )

    assert _fallback_decision(completed_task, "你好").relation == "replace"


@pytest.mark.asyncio
async def test_date_clarification_rebuilds_one_controlled_two_period_query_plan() -> None:
    """LLM 结合待补充任务规整两段范围，后端不自行派生日期。"""
    task = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_METRIC_COMPARISON,
        status=TASK_WAITING_CLARIFICATION,
        route_mode="metrics",
        task_frame={"base_question": "对比本周的 GMV 和上周的 GMV", "supplements": []},
        missing_slots=["本周的起止日期"],
        source_message_ids=["message-1"],
    )

    session = _TaskSession([task, None])
    turn = await prepare_conversation_task(
        session, "conversation-1", "message-2", "本周是指 2026-09-09 至 2026-09-15"
    )
    plan = build_metric_query_plan(turn.effective_question)

    assert turn.requires_clarification is False
    assert task.status == TASK_READY
    assert task.revision == 2
    assert [(unit.start_date, unit.end_date) for unit in plan.units] == [
        (date(2026, 9, 2), date(2026, 9, 8)),
        (date(2026, 9, 9), date(2026, 9, 15)),
    ]
    assert turn.route_override is not None
    assert turn.route_override.mode == "metrics"


def test_knowledge_follow_up_reuses_task_goal_but_requires_fresh_retrieval() -> None:
    """资料续问保留任务目标和补充条件，但不携带上一轮检索结果。"""
    task = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_KNOWLEDGE_QA,
        status=TASK_COMPLETED,
        route_mode="knowledge",
        task_frame={"base_question": "618 活动资料中的发货规则是什么", "supplements": []},
        missing_slots=[],
        source_message_ids=["message-1"],
    )

    turn = _continue_completed_task(task, "message-2", "只看 618 正式期的资料")

    assert task.status == "ready"
    assert "618 活动资料中的发货规则是什么" in turn.effective_question
    assert "只看 618 正式期的资料" in turn.effective_question
    assert turn.route_override is not None
    assert turn.route_override.mode == "knowledge"


def test_registered_activities_resolve_in_user_order() -> None:
    periods = resolve_activity_periods("比较七夕活动和618活动的 GMV")

    assert [label for label, _, _ in periods] == ["七夕", "618"]
    assert [(start, end) for _, start, end in periods] == [
        (date(2026, 8, 10), date(2026, 8, 22)),
        (date(2026, 6, 1), date(2026, 6, 20)),
    ]


@pytest.mark.asyncio
async def test_revise_keeps_confirmed_period_when_current_turn_needs_no_time_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_REVIEW,
        status=TASK_COMPLETED,
        route_mode="hybrid",
        revision=1,
        task_frame={"base_question": "复盘 618 活动", "supplements": []},
        effective_constraints={
            "periods": [{"label": "618", "start_date": "2026-06-01", "end_date": "2026-06-20"}]
        },
        missing_slots=[],
        source_message_ids=["message-1"],
    )

    async def _revise(*_args: object, **_kwargs: object) -> TaskRelationshipDecision:
        return TaskRelationshipDecision("revise", ("analysis_goal",), 0.95, "修改分析重点", "llm")

    async def _clarify(*_args: object, **_kwargs: object) -> TemporalResolution:
        return TemporalResolution("clarify", clarification="请确认日期范围", source="llm")

    monkeypatch.setattr("app.services.conversations.tasks.interpret_task_relationship", _revise)
    monkeypatch.setattr("app.services.conversations.tasks.resolve_temporal_intent", _clarify)
    session = _TaskSession([None, task])

    turn = await prepare_conversation_task(
        session, "conversation-1", "message-2", "继续复盘，重点改为退款商品"
    )

    assert turn.requires_clarification is False
    assert turn.task is task
    assert task.status == TASK_READY
    assert task.revision == 2
    assert task.effective_constraints["periods"] == [
        {"label": "618", "start_date": "2026-06-01", "end_date": "2026-06-20"}
    ]


@pytest.mark.asyncio
async def test_explicitly_preserved_range_cannot_be_replaced_when_goal_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_METRIC_COMPARISON,
        status=TASK_COMPLETED,
        route_mode="metrics",
        revision=1,
        task_frame={"base_question": "对比两段 GMV", "supplements": []},
        effective_constraints={
            "periods": [
                {"label": "本期", "start_date": "2026-09-14", "end_date": "2026-09-16"},
                {"label": "上期", "start_date": "2026-09-07", "end_date": "2026-09-09"},
            ],
            "metrics": ["paid_gmv"],
            "execution_intent": {
                "operation": "comparison",
                "evidence_sources": ["metrics"],
                "output_scope": "comparison_only",
                "allow_metric_drilldown": False,
            },
        },
        missing_slots=[],
        source_message_ids=["message-1"],
    )

    async def _replace(*_args: object, **_kwargs: object) -> TaskRelationshipDecision:
        return TaskRelationshipDecision("replace", ("analysis_goal",), 0.95, "改为归因", "llm")

    async def _resolved(*_args: object, **_kwargs: object) -> TemporalResolution:
        return TemporalResolution(
            "resolved",
            (("错误的新范围", date(2026, 9, 11), date(2026, 9, 17)),),
            source="llm",
        )

    monkeypatch.setattr("app.services.conversations.tasks.interpret_task_relationship", _replace)
    monkeypatch.setattr("app.services.conversations.tasks.resolve_temporal_intent", _resolved)
    session = _TaskSession([None, task])

    turn = await prepare_conversation_task(
        session, "conversation-1", "message-2", "保留刚才的时间范围，解释为什么 GMV 下降"
    )

    assert turn.task is task
    assert task.status == TASK_READY
    assert task.revision == 2
    assert task.effective_constraints["periods"] == [
        {"label": "本期", "start_date": "2026-09-14", "end_date": "2026-09-16"},
        {"label": "上期", "start_date": "2026-09-07", "end_date": "2026-09-09"},
    ]
    assert task.effective_constraints["execution_intent"]["operation"] == "causal_analysis"
    assert task.route_mode == "hybrid"


@pytest.mark.asyncio
async def test_replacement_uses_structured_new_task_input_not_cancelled_task_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_task = ConversationTask(
        conversation_id="conversation-1",
        task_type=TASK_REVIEW,
        status=TASK_COMPLETED,
        route_mode="hybrid",
        revision=1,
        task_frame={"base_question": "复盘 618 活动", "supplements": []},
        effective_constraints={"analysis_goal": "活动复盘", "metrics": ["paid_gmv"]},
        missing_slots=[],
        source_message_ids=["message-1"],
    )
    replacement = "比较七夕和 618 的日均 GMV、支付转化率差异"

    async def _replace(*_args: object, **_kwargs: object) -> TaskRelationshipDecision:
        return TaskRelationshipDecision(
            "replace",
            ("task_type", "metric", "comparison", "time_range"),
            0.95,
            "旧复盘已被新的活动比较替换",
            "llm",
            replacement,
        )

    async def _resolve(_session: object, question: str, **_kwargs: object) -> TemporalResolution:
        if "七夕" in question and "618" in question:
            return TemporalResolution(
                "resolved",
                (
                    ("七夕", date(2026, 8, 10), date(2026, 8, 22)),
                    ("618", date(2026, 6, 1), date(2026, 6, 20)),
                ),
                source="local_gate",
            )
        return TemporalResolution("no_time", source="local_gate")

    monkeypatch.setattr("app.services.conversations.tasks.interpret_task_relationship", _replace)
    monkeypatch.setattr("app.services.conversations.tasks.resolve_temporal_intent", _resolve)
    session = _TaskSession([None, old_task])

    turn = await prepare_conversation_task(
        session,
        "conversation-1",
        "message-2",
        "取消这个复盘，改为比较七夕和 618 的日均 GMV、支付转化率差异。",
    )

    assert old_task.status == "superseded"
    assert turn.task.task_frame["base_question"] == replacement
    assert "analysis_goal" not in turn.task.effective_constraints
    assert turn.task.effective_constraints["execution_intent"] == {
        "operation": "comparison",
        "evidence_sources": ["metrics"],
        "output_scope": "comparison_only",
        "allow_metric_drilldown": False,
    }
