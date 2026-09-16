"""Add persisted metric analysis driver graph.

Revision ID: 20260916_0029
Revises: 20260916_0028
Create Date: 2026-09-16 15:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260916_0029"
down_revision = "20260916_0028"
branch_labels = None
depends_on = None


_DEFAULT_EDGES = (
    ("11000000-0000-0000-0000-000000000001", "paid_gmv", "paid_order_count", 10),
    ("11000000-0000-0000-0000-000000000002", "paid_gmv", "visitor_count", 20),
    ("11000000-0000-0000-0000-000000000003", "paid_gmv", "conversion_rate", 30),
    ("11000000-0000-0000-0000-000000000004", "paid_gmv", "average_order_value", 40),
    ("11000000-0000-0000-0000-000000000005", "paid_order_count", "visitor_count", 10),
    ("11000000-0000-0000-0000-000000000006", "paid_order_count", "conversion_rate", 20),
    ("11000000-0000-0000-0000-000000000007", "refund_rate", "refund_order_count", 10),
    ("11000000-0000-0000-0000-000000000008", "refund_rate", "paid_order_count", 20),
)


def upgrade() -> None:
    op.create_table(
        "metric_analysis_drivers",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "metric_code",
            sa.String(length=100),
            sa.ForeignKey("metric_definitions.metric_code", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "driver_metric_code",
            sa.String(length=100),
            sa.ForeignKey("metric_definitions.metric_code", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relationship_type", sa.String(length=30), nullable=False, server_default="driver"),
        sa.Column("display_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint("metric_code", "driver_metric_code", name="uq_metric_analysis_drivers_edge"),
    )
    op.create_index("ix_metric_analysis_drivers_metric_code", "metric_analysis_drivers", ["metric_code"])
    op.create_index(
        "ix_metric_analysis_drivers_driver_metric_code",
        "metric_analysis_drivers",
        ["driver_metric_code"],
    )

    connection = op.get_bind()
    for edge_id, metric_code, driver_metric_code, display_order in _DEFAULT_EDGES:
        connection.execute(
            sa.text(
                """
                INSERT INTO metric_analysis_drivers
                    (id, metric_code, driver_metric_code, relationship_type, display_order, enabled)
                SELECT CAST(:id AS varchar(36)), CAST(:metric_code AS varchar(100)),
                    CAST(:driver_metric_code AS varchar(100)), 'driver', :display_order, true
                WHERE EXISTS (
                    SELECT 1 FROM metric_definitions
                    WHERE metric_code = CAST(:metric_code AS varchar(100))
                )
                  AND EXISTS (
                    SELECT 1 FROM metric_definitions
                    WHERE metric_code = CAST(:driver_metric_code AS varchar(100))
                )
                ON CONFLICT (metric_code, driver_metric_code) DO NOTHING
                """
            ),
            {
                "id": edge_id,
                "metric_code": metric_code,
                "driver_metric_code": driver_metric_code,
                "display_order": display_order,
            },
        )


def downgrade() -> None:
    op.drop_index("ix_metric_analysis_drivers_driver_metric_code", table_name="metric_analysis_drivers")
    op.drop_index("ix_metric_analysis_drivers_metric_code", table_name="metric_analysis_drivers")
    op.drop_table("metric_analysis_drivers")
