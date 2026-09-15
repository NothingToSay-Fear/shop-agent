"""验证用户手动删除会话时的归属校验和依赖数据清理。"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.conversations import delete_conversation


class FakeSession:
    """仅记录删除语句，避免该接口单元测试依赖真实数据库。"""

    def __init__(self, scalar_results: list[object | None]) -> None:
        self.scalar_results = iter(scalar_results)
        self.statements: list[object] = []
        self.committed = False

    async def scalar(self, _statement: object) -> object | None:
        return next(self.scalar_results)

    async def execute(self, statement: object) -> None:
        self.statements.append(statement)

    async def commit(self) -> None:
        self.committed = True


@pytest.mark.asyncio
async def test_delete_conversation_removes_dependent_session_data() -> None:
    """删除会话必须同时清理消息、短期摘要、候选和运行审计。"""
    session = FakeSession([SimpleNamespace(id="conversation-1"), None])
    user = SimpleNamespace(id="user-1")

    await delete_conversation("conversation-1", session=session, current_user=user)  # type: ignore[arg-type]

    statements = "\n".join(str(statement) for statement in session.statements)
    assert "DELETE FROM tool_calls" in statements
    assert "DELETE FROM user_memory_candidates" in statements
    assert "DELETE FROM conversation_summary_jobs" in statements
    assert "DELETE FROM conversation_summaries" in statements
    assert "DELETE FROM conversation_history_index_jobs" in statements
    assert "DELETE FROM conversation_history_units" in statements
    assert "DELETE FROM agent_runs" in statements
    assert "DELETE FROM messages" in statements
    assert "DELETE FROM conversation_contexts" in statements
    assert "DELETE FROM conversation_tasks" in statements
    assert "DELETE FROM conversations" in statements
    assert session.committed is True


@pytest.mark.asyncio
async def test_delete_conversation_rejects_running_answer() -> None:
    """生成中的会话不能被删除，避免流式请求写入已删除的父记录。"""
    session = FakeSession([SimpleNamespace(id="conversation-1"), "running-run"])
    user = SimpleNamespace(id="user-1")

    with pytest.raises(HTTPException) as error:
        await delete_conversation("conversation-1", session=session, current_user=user)  # type: ignore[arg-type]

    assert error.value.status_code == 409
    assert session.statements == []
    assert session.committed is False


@pytest.mark.asyncio
async def test_delete_conversation_hides_other_users_session() -> None:
    """已知其他用户会话 ID 时仍应返回 404，且不得执行删除。"""
    session = FakeSession([None])
    user = SimpleNamespace(id="user-2")

    with pytest.raises(HTTPException) as error:
        await delete_conversation("conversation-1", session=session, current_user=user)  # type: ignore[arg-type]

    assert error.value.status_code == 404
    assert session.statements == []
