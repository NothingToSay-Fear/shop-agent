"""Add confirmation candidates for conversational long-term memory.

Revision ID: 20260902_0013
Revises: 20260902_0012
Create Date: 2026-09-02 16:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260902_0013"
down_revision = "20260902_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """记录待用户确认的偏好，不把自然语言推断直接写入长期记忆。"""
    op.create_table(
        "user_memory_candidates",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("conversation_id", sa.String(length=36), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_message_id", sa.String(length=36), sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_message_id", sa.String(length=36), sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("memory_type", sa.String(length=30), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=3, scale=2), nullable=False, server_default="0.85"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_user_memory_candidates_user_id", "user_memory_candidates", ["user_id"])
    op.create_index("ix_user_memory_candidates_conversation_id", "user_memory_candidates", ["conversation_id"])
    op.create_index("ix_user_memory_candidates_source_message_id", "user_memory_candidates", ["source_message_id"])
    op.create_index("ix_user_memory_candidates_agent_message_id", "user_memory_candidates", ["agent_message_id"])
    op.create_index("ix_user_memory_candidates_status", "user_memory_candidates", ["status"])


def downgrade() -> None:
    """移除候选确认记录；已确认的 user_memories 不受影响。"""
    op.drop_index("ix_user_memory_candidates_status", table_name="user_memory_candidates")
    op.drop_index("ix_user_memory_candidates_agent_message_id", table_name="user_memory_candidates")
    op.drop_index("ix_user_memory_candidates_source_message_id", table_name="user_memory_candidates")
    op.drop_index("ix_user_memory_candidates_conversation_id", table_name="user_memory_candidates")
    op.drop_index("ix_user_memory_candidates_user_id", table_name="user_memory_candidates")
    op.drop_table("user_memory_candidates")
