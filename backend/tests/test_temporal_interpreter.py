from datetime import date

from app.services.temporal_interpreter import (
    TemporalResolution,
    _needs_llm_from_scores,
    _validate_llm_resolution,
)


def test_llm_normalized_periods_are_accepted_only_with_valid_iso_dates() -> None:
    result = _validate_llm_resolution(
        '{"status":"resolved","periods":['
        '{"label":"上自然周同期","start_date":"2026-09-07","end_date":"2026-09-09"},'
        '{"label":"本自然周累计","start_date":"2026-09-14","end_date":"2026-09-16"}'
        ']}',
        date(2026, 9, 16),
    )

    assert result.status == "resolved"
    assert result.periods == (
        ("上自然周同期", date(2026, 9, 7), date(2026, 9, 9)),
        ("本自然周累计", date(2026, 9, 14), date(2026, 9, 16)),
    )


def test_future_dates_from_llm_are_rejected_before_metric_query() -> None:
    result = _validate_llm_resolution(
        '{"status":"resolved","periods":['
        '{"label":"本月","start_date":"2026-09-01","end_date":"2026-09-30"}'
        ']}',
        date(2026, 9, 16),
    )

    assert result.status == "clarify"
    assert result.source == "invalid"
    assert result.periods == ()


def test_no_time_does_not_inject_a_date_range() -> None:
    result = _validate_llm_resolution('{"status":"no_time","periods":[]}', date(2026, 9, 16))

    assert result.status == "no_time"
    assert result.resets_inherited_range is False


def test_clarification_does_not_reset_an_inherited_date_range() -> None:
    result = TemporalResolution("clarify", clarification="请确认日期范围")

    assert result.resets_inherited_range is False


def test_local_gate_skips_llm_only_for_high_confidence_no_time_turn() -> None:
    assert _needs_llm_from_scores(time_score=0.31, no_time_score=0.71) is False
    assert _needs_llm_from_scores(time_score=0.56, no_time_score=0.60) is True
