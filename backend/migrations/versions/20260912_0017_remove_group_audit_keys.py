"""移除结构化审计数据中的旧分组键。

Revision ID: 20260912_0017
Revises: 20260912_0016
Create Date: 2026-09-12 12:10:00
"""

from alembic import op


revision = "20260912_0017"
down_revision = "20260912_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """清理历史会话字段来源和运行快照中的已废弃分组键。"""
    op.execute(
        """
        UPDATE conversation_contexts
        SET field_sources = (field_sources::jsonb - 'knowledge_group')::json
        WHERE field_sources::jsonb ? 'knowledge_group'
        """
    )
    op.execute(
        """
        UPDATE agent_runs
        SET context_snapshot = (context_snapshot::jsonb - 'knowledge_group')::json
        WHERE context_snapshot::jsonb ? 'knowledge_group'
        """
    )


def downgrade() -> None:
    """已删除的历史分组值无法可靠恢复。"""
