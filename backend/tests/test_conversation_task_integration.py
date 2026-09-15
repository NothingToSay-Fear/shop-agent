"""使用真实 PostgreSQL 经营数据验证跨轮任务补全后的指标 RAG 执行。"""

from datetime import date

import pytest

from app.database import SessionLocal
from app.models import Conversation, User
from app.services.conversation_tasks import prepare_conversation_task
from app.services.metric_rag import MetricQueryConstraints, query_metrics_for_question
from app.services.temporal_interpreter import TemporalResolution

# SessionLocal 复用 asyncpg 连接池；本模块的数据库集成测试必须共用同一事件循环。
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.fixture(autouse=True)
def _controlled_llm_temporal_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """集成测试固定受限 LLM 的分类输出，真实数据库仅验证后端物化和受控 SQL。"""

    async def _resolve(_session: object, question: str, **_kwargs: object) -> TemporalResolution:
        lowered = question.lower()
        if "今天" in lowered:
            return TemporalResolution(
                "resolved", (("今天", date(2026, 9, 15), date(2026, 9, 15)),), source="llm"
            )
        if any(marker in lowered for marker in ("本周", "这周")) and "上周" in lowered:
            return TemporalResolution("clarify", clarification="请确认本周起止日期。", source="llm")
        if "2026-09-09" in question:
            return TemporalResolution(
                "resolved",
                (("上周", date(2026, 9, 2), date(2026, 9, 8)), ("本周", date(2026, 9, 9), date(2026, 9, 15))),
                source="llm",
            )
        if any(marker in lowered for marker in ("上月", "上个月")) and any(
            marker in lowered for marker in ("本月", "这个月")
        ):
            return TemporalResolution(
                "resolved",
                (("上月整月", date(2026, 8, 1), date(2026, 8, 31)), ("本月累计", date(2026, 9, 1), date(2026, 9, 15))),
                source="llm",
            )
        return TemporalResolution("no_time")

    monkeypatch.setattr("app.services.conversation_tasks.resolve_temporal_intent", _resolve)


async def test_task_clarification_requeries_two_weeks_against_real_business_data() -> None:
    """整个事务回滚，验证真实受控 SQL，不向业务库留下测试用户、会话或任务。"""
    async with SessionLocal() as session:
        user = User(
            username="task-integration-test",
            display_name="任务状态集成测试",
            password_hash="not-used",
        )
        session.add(user)
        await session.flush()
        conversation = Conversation(title="任务状态集成测试", user_id=user.id)
        session.add(conversation)
        await session.flush()

        first = await prepare_conversation_task(
            session,
            conversation.id,
            "integration-message-1",
            "对比本周的 GMV 和上周的 GMV",
        )
        assert first.requires_clarification is True

        second = await prepare_conversation_task(
            session,
            conversation.id,
            "integration-message-2",
            "本周是指 2026-09-09 至 2026-09-15",
        )
        context = await query_metrics_for_question(session, second.effective_question)

        assert context is not None
        assert [(item.start_date.isoformat(), item.end_date.isoformat()) for item in context.query_units] == [
            ("2026-09-02", "2026-09-08"),
            ("2026-09-09", "2026-09-15"),
        ]
        assert "【区间对比】" in context.text
        assert "支付 GMV" in context.text
        await session.rollback()


async def test_new_month_comparison_supersedes_waiting_week_task_against_real_business_data() -> None:
    """新月度任务必须关闭旧周任务，并以自然月语义执行上月整月与本月累计查询。"""
    async with SessionLocal() as session:
        user = User(
            username="task-month-integration-test",
            display_name="月度任务状态集成测试",
            password_hash="not-used",
        )
        session.add(user)
        await session.flush()
        conversation = Conversation(title="月度任务状态集成测试", user_id=user.id)
        session.add(conversation)
        await session.flush()

        waiting_week = await prepare_conversation_task(
            session,
            conversation.id,
            "integration-message-1",
            "比较上周和本周的 GMV",
        )
        assert waiting_week.requires_clarification is True

        month_turn = await prepare_conversation_task(
            session,
            conversation.id,
            "integration-message-2",
            "比较一下上个月和这个月的 GMV",
        )
        # 模拟 API 已在上一轮保存过的周范围；新任务生成的两段显式区间必须优先。
        context = await query_metrics_for_question(
            session,
            month_turn.effective_question,
            constraints=MetricQueryConstraints(date(2026, 9, 9), date(2026, 9, 15)),
        )

        assert waiting_week.task.status == "superseded"
        assert month_turn.requires_clarification is False
        assert context is not None
        assert [(item.start_date.isoformat(), item.end_date.isoformat()) for item in context.query_units] == [
            ("2026-08-01", "2026-08-31"),
            ("2026-09-01", "2026-09-15"),
        ]
        assert "【区间对比】" in context.text
        assert "【天数不等提示】" in context.text
        assert "日均支付 GMV" in context.text
        await session.rollback()


async def test_today_metric_query_uses_one_latest_business_day_against_real_data() -> None:
    """时间服务把“今天”转为单日显式范围后，受控 SQL 不得返回默认近 7 天聚合。"""
    async with SessionLocal() as session:
        user = User(username="task-today-integration-test", display_name="时间解释集成测试", password_hash="not-used")
        session.add(user)
        await session.flush()
        conversation = Conversation(title="时间解释集成测试", user_id=user.id)
        session.add(conversation)
        await session.flush()
        turn = await prepare_conversation_task(session, conversation.id, "integration-message-today", "今天的 GMV")
        context = await query_metrics_for_question(session, turn.effective_question)

        assert context is not None
        assert [(item.start_date.isoformat(), item.end_date.isoformat()) for item in context.query_units] == [
            ("2026-09-15", "2026-09-15"),
        ]
        assert "2026-09-15 至 2026-09-15" in context.text
        await session.rollback()
