from datetime import UTC, datetime

from app.agent.execution_plan import build_execution_plan
from app.agent.prompt_builder import PromptBuilder
from app.models import UserMemory
from app.services.intent_router import RetrievalRoute
from app.services.memory.user_memory import (
    MemoryItem,
    UserMemoryContext,
    _default_expiry,
    _parse_remember_content,
    _select_relevant_memories,
    _contains_sensitive_content,
    extract_memory_candidate,
)


def _memory(memory_id: str, memory_type: str, content: str, use_count: int = 0) -> UserMemory:
    """构造不需要数据库连接的记忆记录，用于验证轻量召回规则。"""
    record = UserMemory(
        id=memory_id,
        user_id="user-1",
        memory_type=memory_type,
        content=content,
        use_count=use_count,
    )
    record.updated_at = datetime(2026, 9, 2, tzinfo=UTC)
    return record


def test_memory_candidate_does_not_capture_metric_definition_or_data() -> None:
    """自然表达中的指标口径与业务数据不会成为可确认的长期记忆。"""
    assert extract_memory_candidate("以后 GMV 默认按支付金额统计") is None


def test_memory_candidate_requires_preference_expression() -> None:
    """普通业务提问不会被误判为长期偏好。"""
    assert extract_memory_candidate("分析 618 活动的 GMV") is None
    candidate = extract_memory_candidate("以后复盘时优先按渠道和品类拆分")

    assert candidate is not None
    assert candidate.memory_type == "analysis_preference"
    assert candidate.content == "复盘时优先按渠道和品类拆分"


def test_memory_candidate_supports_stable_work_profile_and_expiring_focus_direction() -> None:
    """工作背景需要确认后长期采用，近期关注方向默认具备有限生命周期。"""
    profile = extract_memory_candidate("我主要负责美妆品类和直播渠道")
    focus = extract_memory_candidate("近期重点关注大促期间的支付转化")

    assert profile is not None
    assert profile.memory_type == "work_profile"
    assert profile.content == "负责美妆品类和直播渠道"
    assert focus is not None
    assert focus.memory_type == "focus_direction"
    assert focus.content == "重点关注大促期间的支付转化"
    assert _default_expiry("work_profile") is None
    assert _default_expiry("focus_direction") is not None


def test_explicit_memory_can_set_an_expiry_date() -> None:
    """用户可用明确日期覆盖近期关注方向的默认有效期。"""
    parsed = _parse_remember_content("到 2099-12-31：近期重点关注直播转化")

    assert parsed.error_message is None
    assert parsed.content == "近期重点关注直播转化"
    assert parsed.expires_at is not None
    assert parsed.expires_at.date().isoformat() == "2099-12-31"


def test_memory_candidate_rejects_sensitive_content() -> None:
    """长期记忆不能将联系方式或凭据类信息沉淀为跨会话上下文。"""
    assert _contains_sensitive_content("联系电话是 13812345678") is True
    assert extract_memory_candidate("我负责团队，邮箱是 ops@example.com") is None


def test_memory_retrieval_prioritizes_answer_and_relevant_analysis_preferences() -> None:
    """分析问题优先采用回答偏好和分析习惯，且最多注入三条。"""
    records = [
        _memory("answer", "answer_preference", "先给结论，再列依据"),
        _memory("analysis", "analysis_preference", "复盘时优先按渠道和品类拆分"),
        _memory("topic", "focus_topic", "关注大促活动复盘"),
        _memory("extra", "focus_topic", "关注新品上架"),
    ]

    selected = _select_relevant_memories(records, "请复盘 618 活动并给出优化建议")

    assert [item.id for item in selected] == ["answer", "analysis", "topic"]


def test_memory_prompt_cannot_override_controlled_query_conditions() -> None:
    """注入提示词必须明确声明长期偏好不能变成业务事实或指标口径。"""
    plan = build_execution_plan(RetrievalRoute("metrics", None, 1.0, False))
    context = UserMemoryContext((MemoryItem("memory-1", "answer_preference", "回答保持简洁"),))

    instruction = PromptBuilder.execution_instruction(
        plan,
        "已验证的工具结果",
        "活动=618；指标=支付 GMV",
        context.display,
    )

    assert "回答偏好：回答保持简洁" in instruction
    assert "不能作为业务事实、数据或指标口径" in instruction
    assert "活动=618；指标=支付 GMV" in instruction
