"""移除未启用的回答反馈表。

Revision ID: 20260822_0004
Revises: 20260821_0003
Create Date: 2026-08-22 10:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260822_0004"
down_revision = "20260821_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """删除未使用的回答反馈表及其索引。"""
    op.drop_index("ix_feedback_message_id", table_name="feedback")
    op.drop_table("feedback")


def downgrade() -> None:
    """回滚时恢复回答反馈表及其索引。"""
    op.create_table(
        "feedback",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("message_id", sa.String(length=36), sa.ForeignKey("messages.id"), nullable=False),
        sa.Column("feedback_type", sa.String(length=20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
    )
    op.create_index("ix_feedback_message_id", "feedback", ["message_id"])
