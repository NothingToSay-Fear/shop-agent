"""扩展用户长期记忆的工作背景与可失效关注方向。

Revision ID: 20260915_0025
Revises: 20260915_0024
Create Date: 2026-09-15 16:30:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260915_0025"
down_revision = "20260915_0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """为待确认的近期关注方向保留有效期，并优化有效记忆读取。"""
    op.add_column(
        "user_memory_candidates",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_user_memory_candidates_expires_at",
        "user_memory_candidates",
        ["expires_at"],
    )
    op.create_index(
        "ix_user_memories_user_status_expires_at",
        "user_memories",
        ["user_id", "status", "expires_at"],
    )


def downgrade() -> None:
    """移除长期记忆有效期候选字段与组合读取索引。"""
    op.drop_index("ix_user_memories_user_status_expires_at", table_name="user_memories")
    op.drop_index("ix_user_memory_candidates_expires_at", table_name="user_memory_candidates")
    op.drop_column("user_memory_candidates", "expires_at")
