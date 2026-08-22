"""增加指标知识与 RAG 检索所需的定义表。

Revision ID: 20260821_0003
Revises: 20260821_0002
Create Date: 2026-08-21 15:30:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260821_0003"
down_revision = "20260821_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """创建指标定义、依赖和向量缓存表。"""
    op.create_table(
        "metric_definitions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("metric_code", sa.String(length=100), nullable=False, unique=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("aliases", sa.JSON(), nullable=False),
        sa.Column("dependency_codes", sa.JSON(), nullable=False),
        sa.Column("query_template", sa.Text(), nullable=True),
        sa.Column("calculation_formula", sa.String(length=100), nullable=True),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("embedding_model", sa.String(length=100), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_index("ix_metric_definitions_enabled", "metric_definitions", ["enabled"])


def downgrade() -> None:
    """删除指标知识表及其索引。"""
    op.drop_index("ix_metric_definitions_enabled", table_name="metric_definitions")
    op.drop_table("metric_definitions")
