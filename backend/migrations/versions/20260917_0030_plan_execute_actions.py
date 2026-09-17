"""Persist executable plan actions and their attempts.

Revision ID: 20260917_0030
Revises: 20260916_0029
Create Date: 2026-09-17 16:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260917_0030"
down_revision = "20260916_0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("task_plans", sa.Column("plan_version", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("task_plans", sa.Column("budget", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")))
    op.add_column("task_plans", sa.Column("planning_context", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")))
    op.alter_column("task_plans", "plan_version", server_default=None)
    op.alter_column("task_plans", "budget", server_default=None)
    op.alter_column("task_plans", "planning_context", server_default=None)

    op.add_column("task_plan_steps", sa.Column("action_type", sa.String(length=50), nullable=False, server_default="tool"))
    op.add_column("task_plan_steps", sa.Column("action_input", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")))
    op.add_column("task_plan_steps", sa.Column("expected_output", sa.Text(), nullable=True))
    op.add_column("task_plan_steps", sa.Column("capability_requirement", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")))
    op.add_column("task_plan_steps", sa.Column("condition", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")))
    op.add_column("task_plan_steps", sa.Column("plan_version", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("task_plan_steps", sa.Column("ordinal", sa.Integer(), nullable=False, server_default="0"))
    for name in ("action_type", "action_input", "capability_requirement", "condition", "plan_version", "ordinal"):
        op.alter_column("task_plan_steps", name, server_default=None)

    op.create_table(
        "task_plan_step_attempts",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("step_id", sa.String(length=36), sa.ForeignKey("task_plan_steps.id", ondelete="CASCADE"), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("input_snapshot", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("evidence_references", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("step_id", "attempt_number", name="uq_task_plan_step_attempt_number"),
    )
    op.create_index("ix_task_plan_step_attempts_step_id", "task_plan_step_attempts", ["step_id"])
    op.create_index("ix_task_plan_step_attempts_status", "task_plan_step_attempts", ["status"])


def downgrade() -> None:
    op.drop_index("ix_task_plan_step_attempts_status", table_name="task_plan_step_attempts")
    op.drop_index("ix_task_plan_step_attempts_step_id", table_name="task_plan_step_attempts")
    op.drop_table("task_plan_step_attempts")
    for name in ("ordinal", "plan_version", "condition", "capability_requirement", "expected_output", "action_input", "action_type"):
        op.drop_column("task_plan_steps", name)
    for name in ("planning_context", "budget", "plan_version"):
        op.drop_column("task_plans", name)
