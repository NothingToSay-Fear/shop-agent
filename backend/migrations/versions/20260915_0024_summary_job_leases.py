"""为会话短期记忆压缩任务增加重试与租约恢复能力。

Revision ID: 20260915_0024
Revises: 20260915_0023
Create Date: 2026-09-15 15:30:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260915_0024"
down_revision = "20260915_0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """使 Worker 崩溃后的压缩任务可在租约到期后重新领取。"""
    op.add_column(
        "conversation_summary_jobs",
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "conversation_summary_jobs",
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
    )
    op.add_column(
        "conversation_summary_jobs",
        sa.Column(
            "run_after",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.add_column(
        "conversation_summary_jobs",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "conversation_summary_jobs",
        sa.Column("lease_token", sa.String(length=36), nullable=True),
    )
    op.create_index("ix_conversation_summary_jobs_run_after", "conversation_summary_jobs", ["run_after"])
    op.create_index(
        "ix_conversation_summary_jobs_lease_expires_at",
        "conversation_summary_jobs",
        ["lease_expires_at"],
    )


def downgrade() -> None:
    """移除压缩任务的租约和重试元数据。"""
    op.drop_index("ix_conversation_summary_jobs_lease_expires_at", table_name="conversation_summary_jobs")
    op.drop_index("ix_conversation_summary_jobs_run_after", table_name="conversation_summary_jobs")
    op.drop_column("conversation_summary_jobs", "lease_token")
    op.drop_column("conversation_summary_jobs", "lease_expires_at")
    op.drop_column("conversation_summary_jobs", "run_after")
    op.drop_column("conversation_summary_jobs", "max_attempts")
    op.drop_column("conversation_summary_jobs", "attempt_count")
