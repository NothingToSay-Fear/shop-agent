import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有 SQLAlchemy 实体共用的基类。"""
    pass


class TimestampMixin:
    """需要记录创建和更新时间的实体共用字段。"""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False,
        comment="最后更新时间",
    )


class User(Base, TimestampMixin):
    """可登录的运营用户；会话与长期记忆均以该实体作为隔离边界。"""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True, comment="唯一登录名")
    display_name: Mapped[str] = mapped_column(String(100), nullable=False, comment="用户展示名称")
    password_hash: Mapped[str] = mapped_column(Text, nullable=False, comment="密码散列，不保存明文密码")
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否具备管理员权限")


class AuthToken(Base):
    """服务端可撤销的登录令牌；数据库仅保存令牌摘要，不保存原始令牌。"""

    __tablename__ = "auth_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联用户标识",
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True, comment="原始令牌摘要，不保存原始令牌")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True, comment="可选失效时间")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        comment="创建时间",
    )


class UserMemory(Base, TimestampMixin):
    """用户主动维护的跨会话偏好，不保存指标口径、业务数据或完整聊天记录。"""

    __tablename__ = "user_memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联用户标识",
    )
    memory_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True, comment="记忆类别")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="正文内容")
    structured_data: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict, comment="可扩展的结构化记忆数据")
    source: Mapped[str] = mapped_column(String(30), nullable=False, default="user_explicit", comment="数据或事件来源")
    confidence: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False, default=Decimal("1.00"), comment="置信度")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active", index=True, comment="当前生命周期或执行状态")
    use_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="被问答采用次数")
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="最近一次被采用时间")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="可选失效时间")


class UserMemoryCandidate(Base):
    """从用户自然表达中提取的候选偏好，确认前不会进入 Agent 上下文。"""

    __tablename__ = "user_memory_candidates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联用户标识",
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联会话标识",
    )
    source_message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联来源消息标识",
    )
    agent_message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联 Agent 消息标识",
    )
    memory_type: Mapped[str] = mapped_column(String(30), nullable=False, comment="记忆类别")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="正文内容")
    confidence: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False, default=Decimal("0.85"), comment="置信度")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True, comment="当前生命周期或执行状态")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True, comment="可选失效时间")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        comment="创建时间",
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="候选被确认或忽略的时间")


class UserMemoryEvent(Base):
    """长期记忆的最小化生命周期审计，不复制记忆正文或原始对话。"""

    __tablename__ = "user_memory_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联用户标识",
    )
    # 不设外键：用户删除记忆后仍保留不含正文的删除审计事件。
    memory_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True, comment="关联记忆标识；审计表不设外键以保留删除记录")
    event_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True, comment="生命周期事件类型")
    source: Mapped[str] = mapped_column(String(30), nullable=False, comment="数据或事件来源")
    candidate_id: Mapped[str | None] = mapped_column(String(36), nullable=True, comment="可选的关联候选标识")
    conversation_id: Mapped[str | None] = mapped_column(String(36), nullable=True, comment="关联会话标识")
    message_id: Mapped[str | None] = mapped_column(String(36), nullable=True, comment="关联消息标识")
    details: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict, comment="附加审计信息")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        comment="创建时间",
    )


class Conversation(Base, TimestampMixin):
    """持久化保存的一条运营讨论会话。"""
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联用户标识",
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="新会话", comment="展示标题")


class ConversationContext(Base, TimestampMixin):
    """会话内已确认的查询条件；不保存聊天原文或模型自行推测的内容。"""

    __tablename__ = "conversation_contexts"

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True,
        comment="关联会话标识",
    )
    activity: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="已确认活动名称")
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True, comment="起始日期")
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True, comment="结束日期")
    metric_hints: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="已确认指标提示列表")
    analysis_goal: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="已确认分析目标")
    field_sources: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict, comment="各约束条件的来源信息")


class ConversationTask(Base, TimestampMixin):
    """会话内可持续补充的任务状态，不将短期记忆或历史回答当作当前查询事实。"""

    __tablename__ = "conversation_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联会话标识",
    )
    task_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True, comment="任务类型，如查询、对比、归因或复盘")
    status: Mapped[str] = mapped_column(String(30), nullable=False, index=True, comment="当前生命周期或执行状态")
    route_mode: Mapped[str | None] = mapped_column(String(30), nullable=True, comment="检索路由模式")
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="修订版本号")
    task_frame: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict, comment="首轮问题、后续补充与解析过程快照")
    effective_constraints: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict, comment="本轮检索、工具调用和生成实际采用的受控约束")
    pending_questions: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="待澄清问题")
    missing_slots: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="尚未确认的结构化约束槽位")
    source_message_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="构成任务的来源消息标识列表")
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="完成时间")


class ConversationTaskEvent(Base):
    """任务状态与约束变更的最小审计事件，不复制完整对话正文。"""

    __tablename__ = "conversation_task_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    task_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_tasks.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联任务标识",
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False, comment="修订版本号")
    event_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True, comment="生命周期事件类型")
    source_message_id: Mapped[str | None] = mapped_column(String(36), nullable=True, comment="关联来源消息标识")
    details: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict, comment="附加审计信息")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        comment="创建时间",
    )


class TaskPlan(Base, TimestampMixin):
    """某个任务版本对应的可审计执行计划。"""

    __tablename__ = "task_plans"
    __table_args__ = (UniqueConstraint("task_id", "revision", name="uq_task_plans_task_revision"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    task_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_tasks.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联任务标识",
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False, comment="修订版本号")
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="ready", index=True, comment="当前生命周期或执行状态")
    summary: Mapped[str] = mapped_column(String(300), nullable=False, comment="摘要内容")
    plan_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="计划内部版本号")
    budget: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict, comment="工具调用与重规划预算")
    planning_context: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict, comment="生成计划时的上下文快照")


class TaskPlanStep(Base, TimestampMixin):
    """计划中的一个受控步骤及其最小执行结果。"""

    __tablename__ = "task_plan_steps"
    __table_args__ = (UniqueConstraint("plan_id", "step_key", name="uq_task_plan_steps_plan_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    plan_id: Mapped[str] = mapped_column(
        ForeignKey("task_plans.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联计划标识",
    )
    step_key: Mapped[str] = mapped_column(String(50), nullable=False, comment="计划内稳定步骤键")
    title: Mapped[str] = mapped_column(String(200), nullable=False, comment="展示标题")
    action_type: Mapped[str] = mapped_column(String(50), nullable=False, default="tool", comment="受控动作类型")
    tool_name: Mapped[str | None] = mapped_column(String(80), nullable=True, comment="调用工具名称")
    depends_on: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="前置步骤键列表")
    action_input: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict, comment="结构化动作输入")
    expected_output: Mapped[str | None] = mapped_column(Text, nullable=True, comment="预期输出说明")
    capability_requirement: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict, comment="允许调用的能力边界")
    condition: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict, comment="条件化执行规则")
    plan_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="计划内部版本号")
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="步骤执行和展示顺序")
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending", index=True, comment="当前生命周期或执行状态")
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True, comment="脱敏结果摘要")
    evidence_references: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="步骤或尝试的证据引用标识列表")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="可空错误摘要")
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="完成时间")


class TaskPlanStepAttempt(Base):
    """一个计划动作的一次执行尝试；步骤可重试或因重规划产生后续动作。"""

    __tablename__ = "task_plan_step_attempts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    step_id: Mapped[str] = mapped_column(
        ForeignKey("task_plan_steps.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联步骤标识",
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False, comment="步骤执行尝试序号")
    status: Mapped[str] = mapped_column(String(30), nullable=False, index=True, comment="当前生命周期或执行状态")
    input_snapshot: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict, comment="本次尝试的输入快照")
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True, comment="脱敏结果摘要")
    evidence_references: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="步骤或尝试的证据引用标识列表")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="可空错误摘要")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        comment="开始处理时间",
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="完成时间")


class ConversationSummary(Base, TimestampMixin):
    """同会话短期状态：压缩摘要与有限最近窗口，不能替代受控检索依据。"""

    __tablename__ = "conversation_summaries"

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True,
        comment="关联会话标识",
    )
    summary_text: Mapped[str] = mapped_column(Text, nullable=False, comment="压缩会话摘要正文")
    topics: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="会话主题列表")
    discussion_points: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="已讨论要点列表")
    open_questions: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="未决问题列表")
    source_message_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="构成任务的来源消息标识列表")
    source_run_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="摘要覆盖的 Agent 运行标识列表")
    recent_turns: Mapped[list[dict[str, str]]] = mapped_column(JSON, nullable=False, default=list, comment="有限最近问答窗口")
    estimated_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="摘要估算 Token 数")
    covered_until_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="摘要覆盖到的时间边界")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="摘要版本号")


class ConversationSummaryJob(Base, TimestampMixin):
    """持久化会话摘要任务；避免在流式回答请求内同步调用模型。"""

    __tablename__ = "conversation_summary_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联会话标识",
    )
    source_agent_message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="触发摘要任务的 Agent 消息标识",
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True, comment="当前生命周期或执行状态")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="已执行或重试次数")
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3, comment="最大重试次数")
    run_after: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True,
        comment="最早可执行时间",
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="可空错误摘要")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="开始处理时间")
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True, comment="Worker 执行租约到期时间")
    lease_token: Mapped[str | None] = mapped_column(String(36), nullable=True, comment="Worker 任务领取租约标识")
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="完成时间")


class ConversationHistoryUnit(Base, TimestampMixin):
    """同会话的一问一答历史单元，供短期状态压缩后的按需召回使用。"""

    __tablename__ = "conversation_history_units"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联会话标识",
    )
    user_message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="触发 Agent 运行的用户消息标识",
    )
    agent_message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, unique=True,
        comment="关联 Agent 消息标识",
    )
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="正文内容")
    search_terms: Mapped[str] = mapped_column(Text, nullable=False, comment="词面检索使用的预处理词项")
    embedding_vector: Mapped[list[float] | None] = mapped_column(Vector(512), nullable=True, comment="语义检索向量")
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="生成向量所使用的模型标识")


class ConversationHistoryIndexJob(Base, TimestampMixin):
    """历史单元向量化任务；词面召回在任务等待期间仍可用。"""

    __tablename__ = "conversation_history_index_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    unit_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_history_units.id", ondelete="CASCADE"), nullable=False, unique=True, index=True,
        comment="关联历史召回单元标识",
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True, comment="当前生命周期或执行状态")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="可空错误摘要")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="开始处理时间")
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="完成时间")


class Product(Base, TimestampMixin):
    """用于内容生成与经营数据关联的商品基础资料。"""

    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    sku: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="唯一商品 SKU")
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="名称")
    category: Mapped[str] = mapped_column(String(100), nullable=False, comment="商品类目")
    price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, comment="精确金额价格")
    highlights: Mapped[str] = mapped_column(Text, nullable=False, comment="商品卖点文本")
    source: Mapped[str] = mapped_column(String(30), nullable=False, default="demo", comment="数据或事件来源")


class DailyMetric(Base):
    """按日期、渠道和商品记录的经营指标，用于趋势与环比分析。"""

    __tablename__ = "daily_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    metric_date: Mapped[date] = mapped_column(Date, nullable=False, index=True, comment="经营指标统计日期")
    channel: Mapped[str] = mapped_column(String(50), nullable=False, index=True, comment="经营渠道")
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"), nullable=False, index=True, comment="关联商品标识")
    visitor_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="访客数量")
    paid_order_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="支付订单数量")
    paid_gmv: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, comment="支付成交金额")
    refund_order_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="退款订单数量")
    source: Mapped[str] = mapped_column(String(30), nullable=False, default="demo", comment="数据或事件来源")


class MetricDefinition(Base, TimestampMixin):
    """可检索的指标知识：业务口径、依赖关系和受控查询模板均由此表管理。"""

    __tablename__ = "metric_definitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    metric_code: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, comment="稳定指标编码")
    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="名称")
    description: Mapped[str] = mapped_column(Text, nullable=False, comment="业务口径描述")
    aliases: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="自然语言别名列表")
    dependency_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="派生指标依赖的指标编码列表")
    query_template: Mapped[str | None] = mapped_column(Text, nullable=True, comment="受控查询模板")
    calculation_formula: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="受控派生计算公式标识")
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True, comment="指标定义语义向量")
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="生成向量所使用的模型标识")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment="是否启用")


class MetricAnalysisDriver(Base, TimestampMixin):
    """指标归因图谱的一条有向边：目标指标由哪些可查询指标驱动。"""

    __tablename__ = "metric_analysis_drivers"
    __table_args__ = (
        UniqueConstraint("metric_code", "driver_metric_code", name="uq_metric_analysis_drivers_edge"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    metric_code: Mapped[str] = mapped_column(
        ForeignKey("metric_definitions.metric_code", ondelete="CASCADE"), nullable=False, index=True,
        comment="稳定指标编码",
    )
    driver_metric_code: Mapped[str] = mapped_column(
        ForeignKey("metric_definitions.metric_code", ondelete="CASCADE"), nullable=False, index=True,
        comment="目标指标的直接驱动指标编码",
    )
    relationship_type: Mapped[str] = mapped_column(String(30), nullable=False, default="driver", comment="指标归因图谱边类型")
    display_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="展示和下钻顺序")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment="是否启用")


class KnowledgeDocument(Base, TimestampMixin):
    """运营资料的原始文件和解析后全文。"""

    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    owner_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="资料所有者用户标识",
    )
    space: Mapped[str] = mapped_column(String(20), nullable=False, default="private", index=True, comment="资料可见空间，如私有或团队")
    title: Mapped[str] = mapped_column(String(200), nullable=False, comment="展示标题")
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False, comment="原始上传文件名")
    file_type: Mapped[str] = mapped_column(String(20), nullable=False, comment="上传文件类型")
    file_path: Mapped[str] = mapped_column(String(500), nullable=False, comment="原始文件持久化路径")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="正文内容")
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="processing", comment="当前生命周期或执行状态")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="可空错误摘要")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="已生成知识切片数量")


class UserKnowledgeDocumentSetting(Base, TimestampMixin):
    """用户对可见资料的检索开关；不改变资料本身的可见范围。"""

    __tablename__ = "user_knowledge_document_settings"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True,
        comment="关联用户标识",
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"), primary_key=True,
        comment="关联文档标识",
    )
    retrieval_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="当前用户是否将资料纳入检索")


class KnowledgeChunk(Base):
    """供语义检索的知识库文本片段及其来源定位信息。"""

    __tablename__ = "knowledge_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联文档标识",
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="文档内切片顺序")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="正文内容")
    # 页码范围与标题路径来自结构化切分，便于资料引用和后续元数据过滤。
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="切片起始页码")
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="切片结束页码")
    heading: Mapped[str | None] = mapped_column(String(300), nullable=True, comment="切片所属标题")
    heading_path: Mapped[str | None] = mapped_column(Text, nullable=True, comment="完整标题层级路径")
    content_type: Mapped[str] = mapped_column(String(20), nullable=False, default="paragraph", comment="切片内容类型")
    # 使用 pgvector 的 HNSW 索引列执行线上语义候选召回。
    embedding_vector: Mapped[list[float] | None] = mapped_column(Vector(512), nullable=True, comment="语义检索向量")
    # 由 jieba 分词后的标题、层级和正文词项，用于 PostgreSQL 全文候选召回。
    search_terms: Mapped[str | None] = mapped_column(Text, nullable=True, comment="词面检索使用的预处理词项")
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="生成向量所使用的模型标识")


class KnowledgeIndexJob(Base, TimestampMixin):
    """持久化的知识库索引任务；Worker 重启后可继续领取未完成工作。"""

    __tablename__ = "knowledge_index_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False, index=True,
        comment="关联文档标识",
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True, comment="当前生命周期或执行状态")
    stage: Mapped[str] = mapped_column(String(30), nullable=False, default="queued", comment="后台任务当前处理阶段")
    processed_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="已处理切片数量")
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="待处理切片总数")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="已执行或重试次数")
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3, comment="最大重试次数")
    run_after: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True,
        comment="最早可执行时间",
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="可空错误摘要")
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="生成向量所使用的模型标识")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="开始处理时间")
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="完成时间")


class Message(Base):
    """会话中的一条用户或 Agent 消息。"""
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True, comment="关联会话标识")
    sender_type: Mapped[str] = mapped_column(String(20), nullable=False, comment="消息发送方类型")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="正文内容")
    data_references: Mapped[str | None] = mapped_column(Text, nullable=True, comment="展示级依据摘要")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="completed", comment="当前生命周期或执行状态")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        comment="创建时间",
    )


class ToolCall(Base):
    """单次 Agent 运行中的受控工具调用审计记录。"""
    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), index=True, comment="关联消息标识")
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=True, index=True,
        comment="关联 Agent 运行标识",
    )
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False, comment="调用工具名称")
    input_summary: Mapped[str | None] = mapped_column(Text, nullable=True, comment="脱敏输入摘要")
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True, comment="脱敏结果摘要")
    reference_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="证据引用标识列表")
    status: Mapped[str] = mapped_column(String(20), nullable=False, comment="当前生命周期或执行状态")
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="调用耗时，单位毫秒")
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="稳定错误码")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        comment="创建时间",
    )


class AgentRun(Base):
    """一次问答的最小化运行审计，不持久化原始问题和原始回答。"""

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="记录唯一标识")
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), nullable=False, index=True, comment="关联会话标识")
    user_message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), nullable=False, index=True, comment="触发 Agent 运行的用户消息标识")
    agent_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.id"), nullable=True, index=True,
        comment="关联 Agent 消息标识",
    )
    question_summary: Mapped[str] = mapped_column(String(300), nullable=False, comment="用户问题摘要，不重复保存完整问题")
    route_mode: Mapped[str | None] = mapped_column(String(30), nullable=True, comment="检索路由模式")
    route_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True, comment="路由置信度")
    route_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否采用保守路由降级")
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True, comment="任务约束采用摘要")
    context_actions: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="上下文采用动作说明")
    context_snapshot: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict, comment="任务约束快照")
    conversation_summary_version: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="采用的会话摘要版本")
    conversation_summary_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否采用会话短期摘要")
    conversation_history_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="采用的历史召回单元标识列表")
    conversation_history_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否采用会话历史召回")
    memory_summary: Mapped[str | None] = mapped_column(Text, nullable=True, comment="长期记忆采用摘要")
    memory_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="采用的长期记忆标识列表")
    memory_selection: Mapped[list[dict[str, str]]] = mapped_column(JSON, nullable=False, default=list, comment="不含正文的记忆选择原因")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running", comment="当前生命周期或执行状态")
    answer_summary: Mapped[str | None] = mapped_column(Text, nullable=True, comment="回答摘要")
    reference_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="证据引用标识列表")
    total_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="总执行耗时，单位毫秒")
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="稳定错误码")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        comment="创建时间",
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="完成时间")
