from datetime import date

from app.services.conversation_context import (
    ConversationContextSnapshot,
    build_context_snapshot,
    build_retrieval_question,
)


def test_context_inherits_activity_period_and_overrides_metric() -> None:
    """后续短问题应继承已确认活动期，但本轮明确指标必须覆盖旧指标。"""
    first = build_context_snapshot(
        ConversationContextSnapshot(),
        "分析 618 活动的 GMV",
        None,
        "message-1",
    )

    second = build_context_snapshot(
        first.snapshot,
        "再看看订单量",
        None,
        "message-2",
    )

    assert second.snapshot.activity == "618"
    assert (second.snapshot.start_date, second.snapshot.end_date) == (
        date(2026, 6, 1),
        date(2026, 6, 20),
    )
    assert second.snapshot.metric_hints == ("支付订单数",)
    assert "活动" in second.inherited_fields
    assert "时间范围" in second.inherited_fields
    assert second.updated_fields == ("指标",)


def test_context_explicit_activity_overrides_previous_activity_and_period() -> None:
    """用户明确切换活动时，不得继续沿用前一活动的时间范围。"""
    previous = ConversationContextSnapshot(
        activity="618",
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 20),
        metric_hints=("支付 GMV",),
    )

    result = build_context_snapshot(previous, "改为七夕活动复盘", None, "message-3")

    assert result.snapshot.activity == "七夕"
    assert (result.snapshot.start_date, result.snapshot.end_date) == (
        date(2026, 8, 10),
        date(2026, 8, 22),
    )
    assert result.snapshot.analysis_goal == "活动复盘"
    assert "活动" in result.updated_fields
    assert "时间范围" in result.updated_fields


def test_context_uses_explicit_dates_over_registered_activity_period() -> None:
    """活动名与明确日期同时出现时，用户写出的日期范围优先。"""
    result = build_context_snapshot(
        ConversationContextSnapshot(),
        "查询 618 在 2026年6月6日至6月18日的 GMV",
        None,
        "message-4",
    )

    assert (result.snapshot.start_date, result.snapshot.end_date) == (
        date(2026, 6, 6),
        date(2026, 6, 18),
    )


def test_context_keeps_selected_knowledge_group_and_builds_controlled_query() -> None:
    """手动选择的资料分组可跨轮继承，并以条件摘要而非伪造提问传给检索层。"""
    first = build_context_snapshot(
        ConversationContextSnapshot(),
        "618 的优惠券规则是什么",
        "618 活动资料",
        "message-5",
    )
    second = build_context_snapshot(first.snapshot, "门槛呢", None, "message-6")

    query = build_retrieval_question("门槛呢", second.snapshot)

    assert second.snapshot.knowledge_group == "618 活动资料"
    assert "资料分组" in second.inherited_fields
    assert query.startswith("门槛呢")
    assert "已确认会话查询条件" in query
    assert "618 活动资料" in query
