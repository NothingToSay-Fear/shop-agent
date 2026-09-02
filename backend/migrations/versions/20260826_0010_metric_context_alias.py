"""补齐会话上下文使用的订单量指标别名。

Revision ID: 20260826_0010
Revises: 20260826_0009
Create Date: 2026-08-26 13:10:00
"""

from alembic import op


revision = "20260826_0010"
down_revision = "20260826_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """仅为内置支付订单数定义补充既有用户常用别名。"""
    op.execute(
        """
        UPDATE metric_definitions
        SET aliases = CASE
            WHEN aliases @> '[\"订单量\"]'::jsonb THEN aliases
            ELSE aliases || '[\"订单量\"]'::jsonb
        END
        WHERE metric_code = 'paid_order_count'
        """
    )


def downgrade() -> None:
    """迁移不删除既有运营人员可能继续使用的别名。"""
