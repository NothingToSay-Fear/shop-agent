"""移除运营待办表。

Revision ID: 20260823_0005
Revises: 20260822_0004
Create Date: 2026-08-23 10:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260823_0005"
down_revision = "20260822_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """删除运营待办表及其中记录。"""
    op.drop_table("tasks")


def downgrade() -> None:
    """回滚时恢复运营待办表结构。"""
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
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
    )
