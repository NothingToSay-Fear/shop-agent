"""增加会话滚动摘要及异步摘要任务。

Revision ID: 20260914_0021
Revises: 20260913_0020
Create Date: 2026-09-14 12:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260914_0021"
down_revision = "20260913_0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """持久化同会话摘要与 Worker 任务，并记录运行时是否采用摘要。"""
    op.create_table(
        "conversation_summaries",
        sa.Column("conversation_id", sa.String(length=36), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("topics", sa.JSON(), nullable=False),
        sa.Column("discussion_points", sa.JSON(), nullable=False),
        sa.Column("open_questions", sa.JSON(), nullable=False),
        sa.Column("source_message_ids", sa.JSON(), nullable=False),
        sa.Column("source_run_ids", sa.JSON(), nullable=False),
        sa.Column("covered_until_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("conversation_id"),
    )
    op.create_table(
        "conversation_summary_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("conversation_id", sa.String(length=36), nullable=False),
        sa.Column("source_agent_message_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_agent_message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversation_summary_jobs_conversation_id", "conversation_summary_jobs", ["conversation_id"])
    op.create_index("ix_conversation_summary_jobs_source_agent_message_id", "conversation_summary_jobs", ["source_agent_message_id"])
    op.create_index("ix_conversation_summary_jobs_status", "conversation_summary_jobs", ["status"])
    op.add_column("agent_runs", sa.Column("conversation_summary_version", sa.Integer(), nullable=True))
    op.add_column(
        "agent_runs",
        sa.Column("conversation_summary_used", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    """删除会话摘要能力；历史消息和会话条件不受影响。"""
    op.drop_column("agent_runs", "conversation_summary_used")
    op.drop_column("agent_runs", "conversation_summary_version")
    op.drop_index("ix_conversation_summary_jobs_status", table_name="conversation_summary_jobs")
    op.drop_index("ix_conversation_summary_jobs_source_agent_message_id", table_name="conversation_summary_jobs")
    op.drop_index("ix_conversation_summary_jobs_conversation_id", table_name="conversation_summary_jobs")
    op.drop_table("conversation_summary_jobs")
    op.drop_table("conversation_summaries")
