"""验证资料空间与用户检索选择的最小安全边界。"""

import pytest

from app.services.knowledge_rag import query_knowledge_for_question


@pytest.mark.asyncio
async def test_knowledge_query_requires_authenticated_user_scope() -> None:
    """没有当前用户 ID 时，知识库查询不能回退为检索全部资料。"""
    result = await query_knowledge_for_question(
        session=None,  # type: ignore[arg-type]
        question="618 优惠券规则是什么？",
        user_id=None,
    )

    assert result is None
