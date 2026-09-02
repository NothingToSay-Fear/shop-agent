"""保存每次运行实际采用的结构化上下文快照。

Revision ID: 20260826_0009
Revises: 20260826_0008
Create Date: 2026-08-26 13:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260826_0009"
down_revision = "20260826_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """让历史运行可复盘其实际采用的字段值，而非只保留展示文案。"""
    op.add_column(
        "agent_runs",
        sa.Column("context_snapshot", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
    )


def downgrade() -> None:
    """移除运行上下文快照。"""
    op.drop_column("agent_runs", "context_snapshot")
