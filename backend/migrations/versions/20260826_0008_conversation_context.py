"""增加会话结构化上下文和本轮上下文审计摘要。

Revision ID: 20260826_0008
Revises: 20260824_0007
Create Date: 2026-08-26 12:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260826_0008"
down_revision = "20260824_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """保存用户明确确认的会话条件，并让运行审计可解释其继承来源。"""
    op.create_table(
        "conversation_contexts",
        sa.Column(
            "conversation_id",
            sa.String(length=36),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("activity", sa.String(length=100), nullable=True),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("metric_hints", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
        sa.Column("knowledge_group", sa.String(length=100), nullable=True),
        sa.Column("analysis_goal", sa.String(length=100), nullable=True),
        sa.Column("field_sources", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
    )
    op.add_column("agent_runs", sa.Column("context_summary", sa.Text(), nullable=True))
    op.add_column(
        "agent_runs",
        sa.Column("context_actions", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
    )


def downgrade() -> None:
    """移除会话上下文结构及其运行摘要。"""
    op.drop_column("agent_runs", "context_actions")
    op.drop_column("agent_runs", "context_summary")
    op.drop_table("conversation_contexts")
