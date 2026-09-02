"""Add local accounts, revocable tokens, and conversation ownership.

Revision ID: 20260902_0011
Revises: 20260826_0010
Create Date: 2026-09-02 12:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260902_0011"
down_revision = "20260826_0010"
branch_labels = None
depends_on = None

_LEGACY_USER_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    """为已有会话保留不可登录的历史归属，并要求新会话必须属于一个用户。"""
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("username", sa.String(length=50), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_table(
        "auth_tokens",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_index("ix_auth_tokens_user_id", "auth_tokens", ["user_id"])
    op.create_index("ix_auth_tokens_token_hash", "auth_tokens", ["token_hash"], unique=True)
    op.create_index("ix_auth_tokens_expires_at", "auth_tokens", ["expires_at"])

    op.add_column("conversations", sa.Column("user_id", sa.String(length=36), nullable=True))
    op.create_foreign_key(
        "fk_conversations_user_id_users", "conversations", "users", ["user_id"], ["id"], ondelete="CASCADE"
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
    op.execute(
        sa.text(
            """
            INSERT INTO users (id, username, display_name, password_hash)
            VALUES (:id, 'legacy_migration_owner', '历史迁移数据', 'disabled')
            ON CONFLICT (id) DO NOTHING
            """
        ).bindparams(id=_LEGACY_USER_ID)
    )
    op.execute(sa.text("UPDATE conversations SET user_id = :id WHERE user_id IS NULL").bindparams(id=_LEGACY_USER_ID))
    op.alter_column("conversations", "user_id", nullable=False)


def downgrade() -> None:
    """仅移除认证与归属字段；历史会话本身不会被删除。"""
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_constraint("fk_conversations_user_id_users", "conversations", type_="foreignkey")
    op.drop_column("conversations", "user_id")
    op.drop_index("ix_auth_tokens_expires_at", table_name="auth_tokens")
    op.drop_index("ix_auth_tokens_token_hash", table_name="auth_tokens")
    op.drop_index("ix_auth_tokens_user_id", table_name="auth_tokens")
    op.drop_table("auth_tokens")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
