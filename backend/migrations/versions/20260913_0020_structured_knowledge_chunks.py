"""为知识库结构化分块保存章节和页码范围。

Revision ID: 20260913_0020
Revises: 20260913_0019
Create Date: 2026-09-13 14:10:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260913_0020"
down_revision = "20260913_0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """迁移旧页码并增加无父子关系的结构化片段元数据。"""
    op.add_column("knowledge_chunks", sa.Column("page_start", sa.Integer(), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("page_end", sa.Integer(), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("heading_path", sa.Text(), nullable=True))
    op.add_column(
        "knowledge_chunks",
        sa.Column("content_type", sa.String(length=20), nullable=False, server_default="paragraph"),
    )
    op.execute(
        "UPDATE knowledge_chunks SET page_start = page_number, page_end = page_number "
        "WHERE page_number IS NOT NULL"
    )
    op.drop_column("knowledge_chunks", "page_number")


def downgrade() -> None:
    """恢复单页码字段；旧切片的标题路径和类型信息会丢失。"""
    op.add_column("knowledge_chunks", sa.Column("page_number", sa.Integer(), nullable=True))
    op.execute("UPDATE knowledge_chunks SET page_number = page_start WHERE page_start IS NOT NULL")
    op.drop_column("knowledge_chunks", "content_type")
    op.drop_column("knowledge_chunks", "heading_path")
    op.drop_column("knowledge_chunks", "page_end")
    op.drop_column("knowledge_chunks", "page_start")
