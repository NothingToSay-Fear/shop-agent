"""Add private/team knowledge spaces and per-user retrieval selections.

Revision ID: 20260911_0015
Revises: 20260911_0014
Create Date: 2026-09-11 21:30:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260911_0015"
down_revision = "20260911_0014"
branch_labels = None
depends_on = None

_LEGACY_USERNAME = "legacy_migration_owner"


def upgrade() -> None:
    """将既有公共资料迁移为团队资料，并建立用户独立的检索选择记录。"""
    op.add_column(
        "users", sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false())
    )
    # 已部署实例没有管理员时，将最早注册的真实账号提升为初始化管理员。
    op.execute(
        sa.text(
            """
            UPDATE users
            SET is_admin = TRUE
            WHERE id = (
                SELECT id FROM users
                WHERE username <> :legacy_username
                ORDER BY created_at, id
                LIMIT 1
            )
            """
        ).bindparams(legacy_username=_LEGACY_USERNAME)
    )

    op.add_column("knowledge_documents", sa.Column("owner_user_id", sa.String(length=36), nullable=True))
    op.add_column(
        "knowledge_documents",
        sa.Column("space", sa.String(length=20), nullable=False, server_default="team"),
    )
    # 既有资料原先是公共资料，因此统一迁入团队空间；优先归属首位真实账号。
    op.execute(
        sa.text(
            """
            UPDATE knowledge_documents
            SET owner_user_id = COALESCE(
                (
                    SELECT id FROM users
                    WHERE username <> :legacy_username
                    ORDER BY created_at, id
                    LIMIT 1
                ),
                (
                    SELECT id FROM users
                    WHERE username = :legacy_username
                    LIMIT 1
                )
            )
            WHERE owner_user_id IS NULL
            """
        ).bindparams(legacy_username=_LEGACY_USERNAME)
    )
    op.alter_column("knowledge_documents", "owner_user_id", nullable=False)
    op.create_foreign_key(
        "fk_knowledge_documents_owner_user_id_users",
        "knowledge_documents",
        "users",
        ["owner_user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_knowledge_documents_owner_user_id", "knowledge_documents", ["owner_user_id"])
    op.create_index("ix_knowledge_documents_space", "knowledge_documents", ["space"])
    op.create_check_constraint(
        "ck_knowledge_documents_space", "knowledge_documents", "space IN ('private', 'team')"
    )

    op.create_table(
        "user_knowledge_document_settings",
        sa.Column(
            "user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column(
            "document_id",
            sa.String(length=36),
            sa.ForeignKey("knowledge_documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("retrieval_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )


def downgrade() -> None:
    """移除资料空间与用户检索选择；不会恢复已删除的资料或片段。"""
    op.drop_table("user_knowledge_document_settings")
    op.drop_constraint("ck_knowledge_documents_space", "knowledge_documents", type_="check")
    op.drop_index("ix_knowledge_documents_space", table_name="knowledge_documents")
    op.drop_index("ix_knowledge_documents_owner_user_id", table_name="knowledge_documents")
    op.drop_constraint(
        "fk_knowledge_documents_owner_user_id_users",
        "knowledge_documents",
        type_="foreignkey",
    )
    op.drop_column("knowledge_documents", "space")
    op.drop_column("knowledge_documents", "owner_user_id")
    op.drop_column("users", "is_admin")
