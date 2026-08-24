"""增加 Agent 运行审计和工具调用关联。

Revision ID: 20260824_0007
Revises: 20260823_0006
Create Date: 2026-08-24 18:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260824_0007"
down_revision = "20260823_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """建立仅保存摘要和引用 ID 的运行审计结构。"""
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("conversation_id", sa.String(length=36), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("user_message_id", sa.String(length=36), sa.ForeignKey("messages.id"), nullable=False),
        sa.Column("agent_message_id", sa.String(length=36), sa.ForeignKey("messages.id"), nullable=True),
        sa.Column("question_summary", sa.String(length=300), nullable=False),
        sa.Column("route_mode", sa.String(length=30), nullable=True),
        sa.Column("route_confidence", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("route_fallback", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="running"),
        sa.Column("answer_summary", sa.Text(), nullable=True),
        sa.Column("reference_ids", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
        sa.Column("total_duration_ms", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_agent_runs_conversation_id", "agent_runs", ["conversation_id"])
    op.create_index("ix_agent_runs_user_message_id", "agent_runs", ["user_message_id"])
    op.create_index("ix_agent_runs_agent_message_id", "agent_runs", ["agent_message_id"])
    op.create_index("ix_agent_runs_created_at", "agent_runs", ["created_at"])

    op.add_column("tool_calls", sa.Column("run_id", sa.String(length=36), nullable=True))
    op.add_column(
        "tool_calls",
        sa.Column("reference_ids", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
    )
    op.create_foreign_key(
        "fk_tool_calls_run_id_agent_runs", "tool_calls", "agent_runs", ["run_id"], ["id"], ondelete="CASCADE"
    )
    op.create_index("ix_tool_calls_run_id", "tool_calls", ["run_id"])


def downgrade() -> None:
    """回滚运行审计结构，不删除原有工具调用明细。"""
    op.drop_index("ix_tool_calls_run_id", table_name="tool_calls")
    op.drop_constraint("fk_tool_calls_run_id_agent_runs", "tool_calls", type_="foreignkey")
    op.drop_column("tool_calls", "reference_ids")
    op.drop_column("tool_calls", "run_id")
    op.drop_index("ix_agent_runs_created_at", table_name="agent_runs")
    op.drop_index("ix_agent_runs_agent_message_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_user_message_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_conversation_id", table_name="agent_runs")
    op.drop_table("agent_runs")
