from datetime import date

from app.services.conversation_context import (
    ConversationContextSnapshot,
    build_context_snapshot,
    build_retrieval_question,
)
from app.services.date_ranges import parse_explicit_date_range


def test_context_inherits_activity_period_and_overrides_metric() -> None:
    """后续短问题继承活动期，但本轮明确指标会覆盖旧指标。"""
    first = build_context_snapshot(
        ConversationContextSnapshot(), "分析 618 活动的 GMV", "message-1"
    )
    second = build_context_snapshot(first.snapshot, "再看看订单量", "message-2")

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
    """用户明确切换活动时，不沿用上一个活动的时间范围。"""
    previous = ConversationContextSnapshot(
        activity="618",
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 20),
        metric_hints=("支付 GMV",),
    )

    result = build_context_snapshot(previous, "改为七夕活动复盘", "message-3")

    assert result.snapshot.activity == "七夕"
    assert (result.snapshot.start_date, result.snapshot.end_date) == (
        date(2026, 8, 10),
        date(2026, 8, 22),
    )
    assert result.snapshot.analysis_goal == "活动复盘"
    assert "活动" in result.updated_fields
    assert "时间范围" in result.updated_fields


def test_context_uses_explicit_dates_over_registered_activity_period() -> None:
    """活动名与明确日期同时出现时，明确日期优先。"""
    result = build_context_snapshot(
        ConversationContextSnapshot(), "查询 618 在 2026年6月6日至6月18日的 GMV", "message-4"
    )

    assert (result.snapshot.start_date, result.snapshot.end_date) == (
        date(2026, 6, 6),
        date(2026, 6, 18),
    )


def test_short_date_range_overrides_old_activity_without_leaving_conflicting_label() -> None:
    """仅指定新日期时，不保留旧活动标签。"""
    previous = ConversationContextSnapshot(
        activity="618",
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 20),
    )

    result = build_context_snapshot(previous, "查询 8/16–8/22 的 GMV", "message-7")

    assert result.snapshot.activity is None
    assert (result.snapshot.start_date, result.snapshot.end_date) == (
        date(2026, 8, 16),
        date(2026, 8, 22),
    )
    assert "活动" in result.cleared_fields


def test_date_range_parser_accepts_common_operator_formats() -> None:
    """上下文与指标服务共享常见日期写法。"""
    assert parse_explicit_date_range("8/16–8/22") == (date(2026, 8, 16), date(2026, 8, 22))
    assert parse_explicit_date_range("8月16日到8月22日") == (date(2026, 8, 16), date(2026, 8, 22))
    assert parse_explicit_date_range("2026-08-16 至 2026-08-22") == (
        date(2026, 8, 16),
        date(2026, 8, 22),
    )


def test_context_can_be_cleared_without_mixing_in_message_history() -> None:
    """重置条件后不遗留范围；会话原文不由条件快照读取。"""
    previous = ConversationContextSnapshot(
        activity="七夕",
        start_date=date(2026, 8, 10),
        end_date=date(2026, 8, 22),
        metric_hints=("支付 GMV",),
    )

    assert "上一轮结论摘要" not in build_retrieval_question("为什么下降？", previous)
    assert "上一轮结论摘要" not in previous.generation_context

    result = build_context_snapshot(previous, "清除会话条件", "message-8")

    assert result.snapshot.display == ""
    assert result.snapshot.metric_hints == ()
