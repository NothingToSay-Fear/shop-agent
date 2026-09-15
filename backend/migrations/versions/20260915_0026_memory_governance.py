"""增加长期记忆生命周期追溯与采用审计。

Revision ID: 20260915_0026
Revises: 20260915_0025
Create Date: 2026-09-15 17:30:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260915_0026"
down_revision = "20260915_0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """以不保存记忆正文的事件表记录创建、确认、替代、过期和删除。"""
    op.create_table(
        "user_memory_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("memory_id", sa.String(length=36), nullable=False),
        sa.Column("event_type", sa.String(length=30), nullable=False),
        sa.Column("source", sa.String(length=30), nullable=False),
        sa.Column("candidate_id", sa.String(length=36), nullable=True),
        sa.Column("conversation_id", sa.String(length=36), nullable=True),
        sa.Column("message_id", sa.String(length=36), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_user_memory_events_user_id", "user_memory_events", ["user_id"])
    op.create_index("ix_user_memory_events_memory_id", "user_memory_events", ["memory_id"])
    op.create_index("ix_user_memory_events_event_type", "user_memory_events", ["event_type"])
    op.add_column("agent_runs", sa.Column("memory_selection", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")))
    op.alter_column("agent_runs", "memory_selection", server_default=None)


def downgrade() -> None:
    """移除记忆治理审计字段。"""
    op.drop_column("agent_runs", "memory_selection")
    op.drop_index("ix_user_memory_events_event_type", table_name="user_memory_events")
    op.drop_index("ix_user_memory_events_memory_id", table_name="user_memory_events")
    op.drop_index("ix_user_memory_events_user_id", table_name="user_memory_events")
    op.drop_table("user_memory_events")
