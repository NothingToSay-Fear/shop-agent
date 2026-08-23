"""增加分组知识库文件和检索片段。

Revision ID: 20260823_0006
Revises: 20260823_0005
Create Date: 2026-08-23 16:45:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260823_0006"
down_revision = "20260823_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """创建原始资料和可向量检索片段表。"""
    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("file_type", sa.String(length=20), nullable=False),
        sa.Column("file_path", sa.String(length=500), nullable=False),
        sa.Column("group_name", sa.String(length=100), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_index("ix_knowledge_documents_group_name", "knowledge_documents", ["group_name"])
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("document_id", sa.String(length=36), sa.ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("heading", sa.String(length=300), nullable=True),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("embedding_model", sa.String(length=100), nullable=True),
    )
    op.create_index("ix_knowledge_chunks_document_id", "knowledge_chunks", ["document_id"])


def downgrade() -> None:
    """回滚时先删除子表，避免外键依赖。"""
    op.drop_index("ix_knowledge_chunks_document_id", table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
    op.drop_index("ix_knowledge_documents_group_name", table_name="knowledge_documents")
    op.drop_table("knowledge_documents")
