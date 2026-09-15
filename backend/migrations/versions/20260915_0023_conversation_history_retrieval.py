"""增加会话内历史召回单元与异步向量化任务。

Revision ID: 20260915_0023
Revises: 20260915_0022
Create Date: 2026-09-15 11:00:00
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "20260915_0023"
down_revision = "20260915_0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """每个已完成问答生成一个可渐进索引的会话历史单元。"""
    op.create_table(
        "conversation_history_units",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("conversation_id", sa.String(length=36), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_message_id", sa.String(length=36), sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_message_id", sa.String(length=36), sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("search_terms", sa.Text(), nullable=False),
        sa.Column("embedding_vector", Vector(512), nullable=True),
        sa.Column("embedding_model", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_conversation_history_units_conversation_id", "conversation_history_units", ["conversation_id"])
    op.create_index("ix_conversation_history_units_user_message_id", "conversation_history_units", ["user_message_id"])
    op.execute(
        """
        CREATE INDEX ix_conversation_history_units_embedding_hnsw
        ON conversation_history_units
        USING hnsw (embedding_vector vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE embedding_vector IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX ix_conversation_history_units_search_terms_fts
        ON conversation_history_units
        USING gin (to_tsvector('simple', search_terms))
        """
    )
    op.create_table(
        "conversation_history_index_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("unit_id", sa.String(length=36), sa.ForeignKey("conversation_history_units.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_conversation_history_index_jobs_status", "conversation_history_index_jobs", ["status"])
    op.add_column("agent_runs", sa.Column("conversation_history_ids", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")))
    op.add_column("agent_runs", sa.Column("conversation_history_used", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.alter_column("agent_runs", "conversation_history_ids", server_default=None)
    op.alter_column("agent_runs", "conversation_history_used", server_default=None)


def downgrade() -> None:
    """移除会话历史召回能力。"""
    op.drop_column("agent_runs", "conversation_history_used")
    op.drop_column("agent_runs", "conversation_history_ids")
    op.drop_index("ix_conversation_history_index_jobs_status", table_name="conversation_history_index_jobs")
    op.drop_table("conversation_history_index_jobs")
    op.drop_index("ix_conversation_history_units_search_terms_fts", table_name="conversation_history_units")
    op.drop_index("ix_conversation_history_units_embedding_hnsw", table_name="conversation_history_units")
    op.drop_index("ix_conversation_history_units_user_message_id", table_name="conversation_history_units")
    op.drop_index("ix_conversation_history_units_conversation_id", table_name="conversation_history_units")
    op.drop_table("conversation_history_units")
