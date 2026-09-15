"""将会话摘要扩展为独立的短期记忆状态。

Revision ID: 20260915_0022
Revises: 20260914_0021
Create Date: 2026-09-15 10:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260915_0022"
down_revision = "20260914_0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """保存有限最近窗口和 token 估算，避免短期记忆反查完整消息表。"""
    op.add_column(
        "conversation_summaries",
        sa.Column("recent_turns", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
    )
    op.add_column(
        "conversation_summaries",
        sa.Column("estimated_tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.alter_column("conversation_summaries", "recent_turns", server_default=None)
    op.alter_column("conversation_summaries", "estimated_tokens", server_default=None)


def downgrade() -> None:
    """移除短期状态窗口字段，保留早期滚动摘要结构。"""
    op.drop_column("conversation_summaries", "estimated_tokens")
    op.drop_column("conversation_summaries", "recent_turns")
