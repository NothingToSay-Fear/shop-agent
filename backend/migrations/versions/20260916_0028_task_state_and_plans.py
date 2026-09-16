"""Add structured task state, events and execution plans.

Revision ID: 20260916_0028
Revises: 20260915_0027
Create Date: 2026-09-16 10:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260916_0028"
down_revision = "20260915_0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversation_tasks",
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "conversation_tasks",
        sa.Column("effective_constraints", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
    )
    op.add_column(
        "conversation_tasks",
        sa.Column("pending_questions", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
    )
    op.alter_column("conversation_tasks", "revision", server_default=None)
    op.alter_column("conversation_tasks", "effective_constraints", server_default=None)
    op.alter_column("conversation_tasks", "pending_questions", server_default=None)

    op.create_table(
        "conversation_task_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(length=36),
            sa.ForeignKey("conversation_tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=30), nullable=False),
        sa.Column("source_message_id", sa.String(length=36), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index("ix_conversation_task_events_task_id", "conversation_task_events", ["task_id"])
    op.create_index("ix_conversation_task_events_event_type", "conversation_task_events", ["event_type"])

    op.create_table(
        "task_plans",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(length=36),
            sa.ForeignKey("conversation_tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="ready"),
        sa.Column("summary", sa.String(length=300), nullable=False),
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
        sa.UniqueConstraint("task_id", "revision", name="uq_task_plans_task_revision"),
    )
    op.create_index("ix_task_plans_task_id", "task_plans", ["task_id"])
    op.create_index("ix_task_plans_status", "task_plans", ["status"])

    op.create_table(
        "task_plan_steps",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "plan_id",
            sa.String(length=36),
            sa.ForeignKey("task_plans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step_key", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("tool_name", sa.String(length=80), nullable=True),
        sa.Column("depends_on", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="pending"),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("evidence_references", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.UniqueConstraint("plan_id", "step_key", name="uq_task_plan_steps_plan_key"),
    )
    op.create_index("ix_task_plan_steps_plan_id", "task_plan_steps", ["plan_id"])
    op.create_index("ix_task_plan_steps_status", "task_plan_steps", ["status"])


def downgrade() -> None:
    op.drop_index("ix_task_plan_steps_status", table_name="task_plan_steps")
    op.drop_index("ix_task_plan_steps_plan_id", table_name="task_plan_steps")
    op.drop_table("task_plan_steps")
    op.drop_index("ix_task_plans_status", table_name="task_plans")
    op.drop_index("ix_task_plans_task_id", table_name="task_plans")
    op.drop_table("task_plans")
    op.drop_index("ix_conversation_task_events_event_type", table_name="conversation_task_events")
    op.drop_index("ix_conversation_task_events_task_id", table_name="conversation_task_events")
    op.drop_table("conversation_task_events")
    op.drop_column("conversation_tasks", "pending_questions")
    op.drop_column("conversation_tasks", "effective_constraints")
    op.drop_column("conversation_tasks", "revision")
