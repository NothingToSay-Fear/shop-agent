"""为知识库候选召回增加 pgvector 与中文全文索引。

Revision ID: 20260913_0018
Revises: 20260912_0017
Create Date: 2026-09-13 11:30:00
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "20260913_0018"
down_revision = "20260912_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """新增可渐进回填的索引列，不删除旧 JSON 向量以保留回滚空间。"""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column(
        "knowledge_chunks",
        sa.Column("embedding_vector", Vector(512), nullable=True),
    )
    op.add_column("knowledge_chunks", sa.Column("search_terms", sa.Text(), nullable=True))
    op.execute(
        """
        CREATE INDEX ix_knowledge_chunks_embedding_vector_hnsw
        ON knowledge_chunks
        USING hnsw (embedding_vector vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE embedding_vector IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX ix_knowledge_chunks_search_terms_fts
        ON knowledge_chunks
        USING gin (to_tsvector('simple', coalesce(search_terms, '')))
        """
    )


def downgrade() -> None:
    """仅移除新增索引列；vector 扩展可能仍被其他对象使用，因此保留。"""
    op.drop_index("ix_knowledge_chunks_search_terms_fts", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunks_embedding_vector_hnsw", table_name="knowledge_chunks")
    op.drop_column("knowledge_chunks", "search_terms")
    op.drop_column("knowledge_chunks", "embedding_vector")
