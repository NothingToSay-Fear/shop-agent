"""移除知识库资料分组。

Revision ID: 20260912_0016
Revises: 20260911_0015
Create Date: 2026-09-12 12:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260912_0016"
down_revision = "20260911_0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """删除资料分组字段及其索引，不再保留历史分组值。"""
    op.drop_index("ix_knowledge_documents_group_name", table_name="knowledge_documents")
    op.drop_column("knowledge_documents", "group_name")
    op.drop_column("conversation_contexts", "knowledge_group")


def downgrade() -> None:
    """为回退旧版本恢复不可为空的未分类字段。"""
    op.add_column(
        "knowledge_documents",
        sa.Column("group_name", sa.String(length=100), nullable=False, server_default="未分类"),
    )
    op.create_index("ix_knowledge_documents_group_name", "knowledge_documents", ["group_name"])
    op.add_column("conversation_contexts", sa.Column("knowledge_group", sa.String(length=100), nullable=True))
