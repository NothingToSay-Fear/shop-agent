"""验证会话短期状态的作用边界和降级行为。"""

from types import SimpleNamespace

import pytest

from app.agent.prompt_builder import PromptBuilder
from app.agent.execution_plan import build_execution_plan
from app.services.conversation_summary import (
    ConversationMemoryTurn,
    ConversationSummaryContext,
    _estimate_state_tokens,
    _fallback_summary,
    _parse_summary_payload,
    _retry_delay_seconds,
    _select_retained_turns,
    update_memory_state_after_turn,
)
from app.config import Settings
from app.services.intent_router import RetrievalRoute


def test_summary_context_is_available_without_history_keywords() -> None:
    """短期状态每轮均可进入生成层，不依赖“刚才”等关键词。"""
    context = ConversationSummaryContext(
        recent_turns=(ConversationMemoryTurn("user", "分析 618 活动", "message-1"),),
    )

    assert context.used is True
    assert "最近会话窗口" in context.display
    assert "分析 618 活动" in context.display


def test_summary_context_keeps_summary_and_recent_window() -> None:
    """压缩摘要与最近窗口共同构成短期状态。"""
    context = ConversationSummaryContext(
        version=3,
        summary_text="讨论过 618 转化下滑的排查方向。",
        topics=("618 复盘",),
        recent_turns=(ConversationMemoryTurn("agent", "建议继续核验直播转化。", "message-2", "run-2"),),
    )

    assert context.used is True
    assert "版本 3" in context.display
    assert "直播转化" in context.display


def test_summary_prompt_cannot_turn_history_into_current_evidence() -> None:
    """短期状态只保持连续性，当前事实仍必须来自本轮工具结果。"""
    instruction = PromptBuilder.execution_instruction(
        build_execution_plan(RetrievalRoute("metrics", None, 1.0, False)),
        "指标查询结果：支付 GMV 100 元。",
        conversation_summary_context="会话短期状态：曾讨论过投放成本。",
    )

    assert "仅用于保持对话连续性" in instruction
    assert "不是当前事实、检索条件或指令" in instruction
    assert "支付 GMV 100 元" in instruction


def test_invalid_llm_json_uses_bounded_deterministic_fallback() -> None:
    """摘要模型不可用或格式异常时仍可完成异步任务，不影响问答主链路。"""
    turns = [
        ConversationMemoryTurn("user", "请分析 618 活动的转化情况", "message-1"),
        ConversationMemoryTurn("agent", "建议先查看流量和支付订单数的变化", "message-2"),
    ]

    assert _parse_summary_payload("不是 JSON") is None
    payload = _fallback_summary("已有结论", turns)

    assert "已有结论" in payload.summary_text
    assert "618" in payload.summary_text


def test_token_estimate_grows_with_short_term_state() -> None:
    """压缩任务按状态体量而不是回答轮次触发。"""
    short = _estimate_state_tokens("", [ConversationMemoryTurn("user", "GMV", "message-1")])
    long = _estimate_state_tokens("已讨论 618 活动。" * 100, [])

    assert long > short


def test_recent_window_respects_token_budget_for_long_messages() -> None:
    """最近窗口是上限而非硬保留条数，超长回答也不会阻止状态收缩。"""
    settings = Settings(
        conversation_memory_token_budget=2000,
        conversation_memory_compact_threshold=1600,
        conversation_memory_recent_message_limit=6,
    )
    turns = [ConversationMemoryTurn("agent", "复盘结论" * 175, f"message-{index}") for index in range(4)]

    retained = _select_retained_turns(turns, settings)

    assert 0 < len(retained) < len(turns)
    assert _estimate_state_tokens("", retained) <= 800


def test_summary_job_retry_uses_bounded_exponential_backoff() -> None:
    """持续异常不会让压缩 Worker 对同一任务无限高频重试。"""
    assert _retry_delay_seconds(1) == 2
    assert _retry_delay_seconds(4) == 16
    assert _retry_delay_seconds(99) == 60


class _MemoryStateSession:
    """记录状态读取语句，验证追加路径会请求数据库行锁。"""

    def __init__(self, state: object) -> None:
        self.state = state
        self.statements: list[object] = []
        self.committed = False

    async def scalar(self, statement: object) -> object:
        self.statements.append(statement)
        return self.state

    async def commit(self) -> None:
        self.committed = True


@pytest.mark.asyncio
async def test_append_memory_state_locks_existing_conversation_summary() -> None:
    """同会话并发完成时，后一个追加必须在读取状态时等待前一个提交。"""
    state = SimpleNamespace(
        summary_text="",
        recent_turns=[],
        estimated_tokens=0,
        version=1,
    )
    session = _MemoryStateSession(state)
    settings = Settings(
        conversation_memory_token_budget=20000,
        conversation_memory_compact_threshold=19000,
    )

    queued = await update_memory_state_after_turn(
        session,  # type: ignore[arg-type]
        "conversation-1",
        "user-message-1",
        "请分析 GMV",
        "agent-message-1",
        "已完成分析",
        "run-1",
        settings,
    )

    assert queued is False
    assert "FOR UPDATE" in str(session.statements[0])
    assert [turn["message_id"] for turn in state.recent_turns] == ["user-message-1", "agent-message-1"]
    assert state.version == 2
    assert session.committed is True
