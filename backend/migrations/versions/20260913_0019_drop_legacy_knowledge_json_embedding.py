"""移除已完成迁移的知识库 JSON 向量列。

Revision ID: 20260913_0019
Revises: 20260913_0018
Create Date: 2026-09-13 12:10:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260913_0019"
down_revision = "20260913_0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """删除旧 JSON 向量；pgvector 向量和词面索引已完成回填。"""
    op.drop_column("knowledge_chunks", "embedding")


def downgrade() -> None:
    """仅恢复列结构，已删除的旧 JSON 向量数据不可恢复。"""
    op.add_column("knowledge_chunks", sa.Column("embedding", sa.JSON(), nullable=True))
