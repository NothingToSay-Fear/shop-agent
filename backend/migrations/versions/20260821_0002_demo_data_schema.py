"""增加商品和每日经营指标表。

Revision ID: 20260821_0002
Revises: 20260821_0001
Create Date: 2026-08-21 12:30:00
"""

from alembic import op
import sqlalchemy as sa

revision = "20260821_0002"
down_revision = "20260821_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """创建商品资料和按日汇总经营数据所需的数据表。"""
    op.create_table(
        "products",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("sku", sa.String(length=50), nullable=False, unique=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("price", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("highlights", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_table(
        "daily_metrics",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("metric_date", sa.Date(), nullable=False),
        sa.Column("channel", sa.String(length=50), nullable=False),
        sa.Column("product_id", sa.String(length=36), sa.ForeignKey("products.id"), nullable=False),
        sa.Column("visitor_count", sa.Integer(), nullable=False),
        sa.Column("paid_order_count", sa.Integer(), nullable=False),
        sa.Column("paid_gmv", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("refund_order_count", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=30), nullable=False),
    )
    op.create_index("ix_daily_metrics_metric_date", "daily_metrics", ["metric_date"])
    op.create_index("ix_daily_metrics_channel", "daily_metrics", ["channel"])
    op.create_index("ix_daily_metrics_product_id", "daily_metrics", ["product_id"])


def downgrade() -> None:
    """按依赖关系删除模拟经营数据表和商品资料表。"""
    op.drop_index("ix_daily_metrics_product_id", table_name="daily_metrics")
    op.drop_index("ix_daily_metrics_channel", table_name="daily_metrics")
    op.drop_index("ix_daily_metrics_metric_date", table_name="daily_metrics")
    op.drop_table("daily_metrics")
    op.drop_table("products")
