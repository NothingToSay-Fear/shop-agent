"""增加可跨轮补充的会话任务状态。

Revision ID: 20260915_0027
Revises: 20260915_0026
Create Date: 2026-09-15 18:30:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260915_0027"
down_revision = "20260915_0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """持久化待补充任务，避免把后续澄清语句误判为独立问题。"""
    op.create_table(
        "conversation_tasks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(length=36),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("task_type", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("route_mode", sa.String(length=30), nullable=True),
        sa.Column("task_frame", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("missing_slots", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
        sa.Column("source_message_ids", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index("ix_conversation_tasks_conversation_id", "conversation_tasks", ["conversation_id"])
    op.create_index("ix_conversation_tasks_task_type", "conversation_tasks", ["task_type"])
    op.create_index("ix_conversation_tasks_status", "conversation_tasks", ["status"])


def downgrade() -> None:
    """移除会话任务状态。"""
    op.drop_index("ix_conversation_tasks_status", table_name="conversation_tasks")
    op.drop_index("ix_conversation_tasks_task_type", table_name="conversation_tasks")
    op.drop_index("ix_conversation_tasks_conversation_id", table_name="conversation_tasks")
    op.drop_table("conversation_tasks")
