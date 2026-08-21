"""Create initial Shop Agent schema.

Revision ID: 20260821_0001
Revises:
Create Date: 2026-08-21 12:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = "20260821_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 按依赖顺序创建 MVP 所需的持久化实体。
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("conversation_id", sa.String(length=36), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("sender_type", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("data_references", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_table(
        "tasks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("source_message_id", sa.String(length=36), sa.ForeignKey("messages.id"), nullable=True),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("assignee", sa.String(length=100), nullable=True),
        sa.Column("due_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("priority", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("acceptance_metric", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_table(
        "feedback",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("message_id", sa.String(length=36), sa.ForeignKey("messages.id"), nullable=False),
        sa.Column("feedback_type", sa.String(length=20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_index("ix_feedback_message_id", "feedback", ["message_id"])
    op.create_table(
        "tool_calls",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("message_id", sa.String(length=36), sa.ForeignKey("messages.id"), nullable=False),
        sa.Column("tool_name", sa.String(length=100), nullable=False),
        sa.Column("input_summary", sa.Text(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_index("ix_tool_calls_message_id", "tool_calls", ["message_id"])


def downgrade() -> None:
    # 先删除依赖表和索引，再删除被引用的父表。
    op.drop_index("ix_tool_calls_message_id", table_name="tool_calls")
    op.drop_table("tool_calls")
    op.drop_index("ix_feedback_message_id", table_name="feedback")
    op.drop_table("feedback")
    op.drop_table("tasks")
    op.drop_index("ix_messages_conversation_id", table_name="messages")
    op.drop_table("messages")
    op.drop_table("conversations")
