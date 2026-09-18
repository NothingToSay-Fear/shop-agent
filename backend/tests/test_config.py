"""验证会话记忆配置的默认值与边界关系。"""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_conversation_memory_defaults_support_longer_tasks() -> None:
    settings = Settings()

    assert settings.conversation_memory_token_budget == 12000
    assert settings.conversation_memory_compact_threshold == 9600
    assert settings.conversation_memory_recent_message_limit == 10
    assert settings.conversation_memory_turn_text_limit == 2000
    assert settings.conversation_summary_llm_text_limit == 1200
    assert settings.conversation_summary_fallback_text_limit == 2400


def test_conversation_memory_threshold_must_leave_compaction_headroom() -> None:
    with pytest.raises(ValidationError, match="conversation_memory_compact_threshold"):
        Settings(
            conversation_memory_token_budget=12000,
            conversation_memory_compact_threshold=12000,
        )


def test_llm_summary_limit_cannot_exceed_fallback_limit() -> None:
    with pytest.raises(ValidationError, match="conversation_summary_llm_text_limit"):
        Settings(
            conversation_summary_llm_text_limit=1600,
            conversation_summary_fallback_text_limit=1200,
        )
