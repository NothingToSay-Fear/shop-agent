"""Add user-scoped long-term memories and memory usage audit fields.

Revision ID: 20260902_0012
Revises: 20260902_0011
Create Date: 2026-09-02 15:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260902_0012"
down_revision = "20260902_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """建立按用户隔离、可停用和可审计的长期记忆，不存放指标口径。"""
    op.create_table(
        "user_memories",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("memory_type", sa.String(length=30), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("structured_data", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("source", sa.String(length=30), nullable=False, server_default="user_explicit"),
        sa.Column("confidence", sa.Numeric(precision=3, scale=2), nullable=False, server_default="1.00"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("use_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_user_memories_user_id", "user_memories", ["user_id"])
    op.create_index("ix_user_memories_memory_type", "user_memories", ["memory_type"])
    op.create_index("ix_user_memories_status", "user_memories", ["status"])
    op.add_column("agent_runs", sa.Column("memory_summary", sa.Text(), nullable=True))
    op.add_column(
        "agent_runs",
        sa.Column("memory_ids", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
    )


def downgrade() -> None:
    """回滚长期记忆与运行审计字段，不影响用户、会话和消息。"""
    op.drop_column("agent_runs", "memory_ids")
    op.drop_column("agent_runs", "memory_summary")
    op.drop_index("ix_user_memories_status", table_name="user_memories")
    op.drop_index("ix_user_memories_memory_type", table_name="user_memories")
    op.drop_index("ix_user_memories_user_id", table_name="user_memories")
    op.drop_table("user_memories")
