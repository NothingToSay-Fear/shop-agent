"""Add durable background indexing jobs for knowledge documents.

Revision ID: 20260911_0014
Revises: 20260902_0013
Create Date: 2026-09-11 20:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260911_0014"
down_revision = "20260902_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """创建可恢复、可重试的资料索引任务表。"""
    op.create_table(
        "knowledge_index_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("document_id", sa.String(length=36), sa.ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="queued"),
        sa.Column("stage", sa.String(length=30), nullable=False, server_default="queued"),
        sa.Column("processed_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("run_after", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("embedding_model", sa.String(length=100), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_knowledge_index_jobs_document_id", "knowledge_index_jobs", ["document_id"])
    op.create_index("ix_knowledge_index_jobs_status", "knowledge_index_jobs", ["status"])
    op.create_index("ix_knowledge_index_jobs_run_after", "knowledge_index_jobs", ["run_after"])


def downgrade() -> None:
    """删除索引任务表。"""
    op.drop_index("ix_knowledge_index_jobs_run_after", table_name="knowledge_index_jobs")
    op.drop_index("ix_knowledge_index_jobs_status", table_name="knowledge_index_jobs")
    op.drop_index("ix_knowledge_index_jobs_document_id", table_name="knowledge_index_jobs")
    op.drop_table("knowledge_index_jobs")
