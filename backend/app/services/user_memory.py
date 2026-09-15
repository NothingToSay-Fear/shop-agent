"""按用户召回、确认和维护跨会话长期偏好。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
import logging
import re

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import UserMemory, UserMemoryCandidate, UserMemoryEvent

logger = logging.getLogger("uvicorn.error")

MAX_MEMORIES_PER_QUERY = 3
_ANALYSIS_KEYWORDS = ("分析", "复盘", "对比", "诊断", "原因", "建议", "优化")
_FORBIDDEN_MEMORY_KEYWORDS = (
    "gmv",
    "订单量",
    "支付金额",
    "客单价",
    "转化率",
    "退款率",
    "指标口径",
    "指标",
)
_REMEMBER_PREFIX = re.compile(r"^(?:请|帮我)?记住[：:，,\s]*(.+)$")
_FORGET_PREFIX = re.compile(r"^(?:请|帮我)?忘记[：:，,\s]*(.+)$")
_CANDIDATE_PATTERNS = (
    re.compile(r"^(?:以后|今后|后续)(?:请|希望)?[，,\s]*(.+)$"),
    re.compile(r"^(?:我)?希望以后[，,\s]*(.+)$"),
    re.compile(r"^我(?:通常|习惯)[，,\s]*(.+)$"),
)
_WORK_PROFILE_PATTERNS = (
    re.compile(r"^(?:我)?(?:主要)?负责[：:，,\s]*(.+)$"),
    re.compile(r"^(?:我)?(?:主要)?做[：:，,\s]*(.+)$"),
)
_FOCUS_DIRECTION_PATTERNS = (
    re.compile(r"^(?:近期|当前|目前|这段时间)(?:我)?(?:重点)?关注[：:，,\s]*(.+)$"),
    re.compile(r"^(?:我)?(?:近期|当前|目前)的?(?:重点|主要)是[：:，,\s]*(.+)$"),
)
_EXPIRES_ON_PREFIX = re.compile(r"^(?:到|截至|有效至)[：:，,\s]*(\d{4}-\d{1,2}-\d{1,2})[：:，,\s]*(.+)$")
_UPDATE_PREFIX = re.compile(r"^(?:请|帮我)?更新(?:(工作背景|近期关注|分析习惯|回答偏好))?[：:，,\s]*(.+)$")
_SENSITIVE_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE),
    re.compile(r"(?<![\w\d])\d{17}[\dXx](?![\w\d])"),
    re.compile(r"\b(?:sk|api)[-_][A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
)
_SENSITIVE_KEYWORDS = ("密码", "密钥", "token", "身份证", "银行卡")
_MEMORY_TYPE_BY_LABEL = {
    "工作背景": "work_profile",
    "近期关注": "focus_direction",
    "分析习惯": "analysis_preference",
    "回答偏好": "answer_preference",
}


@dataclass(frozen=True)
class MemoryItem:
    """传给 Agent 的最小化长期记忆，不携带内部状态或用户标识。"""

    id: str
    memory_type: str
    content: str
    selection_reason: str = ""


@dataclass(frozen=True)
class UserMemoryContext:
    """一次提问实际采用的用户级长期记忆。"""

    items: tuple[MemoryItem, ...] = ()

    @property
    def ids(self) -> list[str]:
        return [item.id for item in self.items]

    @property
    def display(self) -> str:
        """为生成层提供边界明确的偏好提示，而非额外业务事实。"""
        if not self.items:
            return ""
        labels = {
            "analysis_preference": "分析习惯",
            "answer_preference": "回答偏好",
            "focus_topic": "关注主题",
            "work_profile": "工作背景",
            "focus_direction": "近期关注",
        }
        return "\n".join(
            f"- {labels.get(item.memory_type, '用户偏好')}：{item.content}" for item in self.items
        )

    @property
    def audit_summary(self) -> str | None:
        if not self.items:
            return None
        return f"已采用 {len(self.items)} 条用户长期记忆"

    @property
    def audit_selection(self) -> list[dict[str, str]]:
        """审计只保存类型与选择原因，不复制用户记忆正文。"""
        return [
            {"memory_id": item.id, "memory_type": item.memory_type, "reason": item.selection_reason}
            for item in self.items
        ]


@dataclass(frozen=True)
class MemoryCommandResult:
    """显式记忆命令的确定性响应，不触发 Agent、RAG 或业务查询。"""

    message: str


@dataclass(frozen=True)
class MemoryCandidateProposal:
    """从自然表达提取、但尚未确认的偏好候选。"""

    memory_type: str
    content: str


@dataclass(frozen=True)
class _RememberContent:
    """显式记忆命令解析后的正文与可选失效时间。"""

    content: str
    expires_at: datetime | None = None
    error_message: str | None = None


class MemoryService:
    """将可确认的偏好压缩为每轮最多三条的 Agent 上下文。"""

    @staticmethod
    async def retrieve_for_query(
        session: AsyncSession, user_id: str, question: str
    ) -> UserMemoryContext:
        """只读取当前用户有效记忆，并按问题场景选择少量偏好。"""
        now = datetime.now(UTC)
        expired_count = await _expire_active_memories(session, user_id, now)
        records = list(
            await session.scalars(
                select(UserMemory)
                .where(
                    UserMemory.user_id == user_id,
                    UserMemory.status == "active",
                    or_(UserMemory.expires_at.is_(None), UserMemory.expires_at > now),
                )
                .order_by(UserMemory.updated_at.desc())
            )
        )
        selected = _select_relevant_memories(records, question)
        context = UserMemoryContext(
            tuple(
                MemoryItem(item.id, item.memory_type, item.content, _selection_reason(item, question))
                for item in selected
            )
        )
        logger.info(
            "user_memory_retrieved user_id=%s active_records=%s selected_records=%s expired_records=%s",
            user_id,
            len(records),
            len(selected),
            expired_count,
        )
        return context

    @staticmethod
    def record_usage(records: list[UserMemory]) -> None:
        """在与本轮消息相同事务内记录实际采用的记忆。"""
        now = datetime.now(UTC)
        for record in records:
            record.use_count += 1
            record.last_used_at = now

    @staticmethod
    async def record_context_usage(
        session: AsyncSession, user_id: str, context: UserMemoryContext
    ) -> None:
        """只更新本轮选中的记录，不扫描或修改其他用户的记忆。"""
        if not context.items:
            return
        records = list(
            await session.scalars(
                select(UserMemory).where(
                    UserMemory.id.in_(context.ids), UserMemory.user_id == user_id
                )
            )
        )
        MemoryService.record_usage(records)

    @staticmethod
    async def handle_explicit_command(
        session: AsyncSession,
        user_id: str,
        question: str,
        conversation_id: str | None = None,
        source_message_id: str | None = None,
    ) -> MemoryCommandResult | None:
        """识别“记住、忘记、清除所有记忆”等显式命令并在用户域内执行。"""
        normalized = question.strip()
        now = datetime.now(UTC)
        await _expire_active_memories(session, user_id, now)
        if normalized in {"清除所有记忆", "清空所有记忆", "忘记所有记忆"}:
            records = list(await session.scalars(select(UserMemory).where(UserMemory.user_id == user_id)))
            for record in records:
                _append_memory_event(session, record, "deleted", "user_explicit", conversation_id, source_message_id)
            await session.execute(delete(UserMemory).where(UserMemory.user_id == user_id))
            await session.execute(
                update(UserMemoryCandidate)
                .where(
                    UserMemoryCandidate.user_id == user_id,
                    UserMemoryCandidate.status == "pending",
                )
                .values(status="dismissed", resolved_at=datetime.now(UTC))
            )
            await session.commit()
            return MemoryCommandResult("已清除你的长期记忆和待确认候选；历史会话与审计记录不会受影响。")

        update_match = _UPDATE_PREFIX.match(normalized)
        if update_match:
            memory_type = _MEMORY_TYPE_BY_LABEL.get(update_match.group(1) or "")
            parsed = _parse_remember_content(update_match.group(2))
            if parsed.error_message:
                return MemoryCommandResult(parsed.error_message)
            content = parsed.content
            if _contains_sensitive_content(content):
                return MemoryCommandResult("长期记忆不能保存手机号、邮箱、证件号、密码、密钥或令牌等敏感信息。")
            if _contains_forbidden_content(content):
                return MemoryCommandResult("长期记忆不能保存指标口径或业务数据。")
            memory_type = memory_type or _classify_memory_type(content)
            return await _replace_memory_type(
                session, user_id, memory_type, content, parsed.expires_at or _default_expiry(memory_type),
                conversation_id, source_message_id,
            )

        remember_match = _REMEMBER_PREFIX.match(normalized)
        if remember_match:
            parsed = _parse_remember_content(remember_match.group(1))
            if parsed.error_message:
                return MemoryCommandResult(parsed.error_message)
            content = parsed.content
            if _contains_sensitive_content(content):
                return MemoryCommandResult("长期记忆不能保存手机号、邮箱、证件号、密码、密钥或令牌等敏感信息。")
            if _contains_forbidden_content(content):
                return MemoryCommandResult("长期记忆只保存回答偏好、分析习惯、工作背景或关注方向，不能保存指标口径或业务数据。")
            if len(content) < 2:
                return MemoryCommandResult("请在“记住”后补充至少两个字符的偏好内容。")
            memory_type = _classify_memory_type(content)
            expires_at = parsed.expires_at or _default_expiry(memory_type)
            existing = await session.scalar(
                select(UserMemory).where(
                    UserMemory.user_id == user_id,
                    UserMemory.memory_type == memory_type,
                    UserMemory.content == content,
                )
            )
            if existing is None:
                if await _active_memory_count(session, user_id) >= get_settings().user_memory_max_active_records:
                    return MemoryCommandResult("你的有效长期记忆已达到上限，请先使用“忘记 …”清理不再需要的内容。")
                existing = UserMemory(
                        user_id=user_id,
                        memory_type=memory_type,
                        content=content,
                        source="user_explicit",
                        status="active",
                        expires_at=expires_at,
                    )
                session.add(existing)
                await session.flush()
                _append_memory_event(session, existing, "created", "user_explicit", conversation_id, source_message_id)
            else:
                existing.status = "active"
                existing.source = "user_explicit"
                existing.expires_at = expires_at
                _append_memory_event(session, existing, "reaffirmed", "user_explicit", conversation_id, source_message_id)
            await session.commit()
            expires_display = f"（有效至 {expires_at.date().isoformat()}）" if expires_at else ""
            return MemoryCommandResult(f"已记住：{content}{expires_display}")

        forget_match = _FORGET_PREFIX.match(normalized)
        if forget_match:
            content = _normalize_memory_content(forget_match.group(1))
            if len(content) < 2:
                return MemoryCommandResult("请在“忘记”后说明需要清除的偏好内容。")
            records = list(
                await session.scalars(
                    select(UserMemory).where(
                        UserMemory.user_id == user_id,
                        UserMemory.content.contains(content),
                    )
                )
            )
            for record in records:
                _append_memory_event(session, record, "deleted", "user_explicit", conversation_id, source_message_id)
                await session.delete(record)
            await session.commit()
            if records:
                return MemoryCommandResult(f"已忘记与“{content}”相关的 {len(records)} 条长期记忆。")
            return MemoryCommandResult(f"没有找到与“{content}”相关的长期记忆。")

        return None

    @staticmethod
    async def create_candidate_if_eligible(
        session: AsyncSession,
        user_id: str,
        conversation_id: str,
        source_message_id: str,
        agent_message_id: str,
        question: str,
    ) -> UserMemoryCandidate | None:
        """在回答落库后生成候选；重复偏好不重复提示，且候选不会参与本轮回答。"""
        proposal = extract_memory_candidate(question)
        if proposal is None:
            return None
        now = datetime.now(UTC)
        await _expire_active_memories(session, user_id, now)
        if await _active_memory_count(session, user_id) >= get_settings().user_memory_max_active_records:
            logger.info("user_memory_candidate_skipped user_id=%s reason=capacity", user_id)
            return None
        existing_memory = await session.scalar(
            select(UserMemory.id).where(
                UserMemory.user_id == user_id,
                UserMemory.memory_type == proposal.memory_type,
                UserMemory.content == proposal.content,
                UserMemory.status == "active",
                or_(UserMemory.expires_at.is_(None), UserMemory.expires_at > now),
            )
        )
        if existing_memory is not None:
            return None
        existing_candidate = await session.scalar(
            select(UserMemoryCandidate.id).where(
                UserMemoryCandidate.user_id == user_id,
                UserMemoryCandidate.memory_type == proposal.memory_type,
                UserMemoryCandidate.content == proposal.content,
                UserMemoryCandidate.status == "pending",
                or_(UserMemoryCandidate.expires_at.is_(None), UserMemoryCandidate.expires_at > now),
            )
        )
        if existing_candidate is not None:
            return None
        candidate = UserMemoryCandidate(
            user_id=user_id,
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            agent_message_id=agent_message_id,
            memory_type=proposal.memory_type,
            content=proposal.content,
            confidence=0.85,
            expires_at=_default_expiry(proposal.memory_type),
        )
        session.add(candidate)
        await session.commit()
        await session.refresh(candidate)
        return candidate

    @staticmethod
    async def accept_candidate(session: AsyncSession, candidate: UserMemoryCandidate) -> None:
        """把已确认候选转化为用户记忆；同内容重复确认时复用已有记录。"""
        await _expire_active_memories(session, candidate.user_id, datetime.now(UTC))
        memory = await session.scalar(
            select(UserMemory).where(
                UserMemory.user_id == candidate.user_id,
                UserMemory.memory_type == candidate.memory_type,
                UserMemory.content == candidate.content,
            )
        )
        if memory is None:
            if await _active_memory_count(session, candidate.user_id) >= get_settings().user_memory_max_active_records:
                raise ValueError("你的有效长期记忆已达到上限，请先清理不再需要的内容。")
            memory = UserMemory(
                    user_id=candidate.user_id,
                    memory_type=candidate.memory_type,
                    content=candidate.content,
                    source="conversation_confirmed",
                    confidence=candidate.confidence,
                    status="active",
                    expires_at=candidate.expires_at,
                )
            session.add(memory)
            await session.flush()
        else:
            memory.status = "active"
            memory.source = "conversation_confirmed"
            memory.expires_at = candidate.expires_at
        candidate.status = "accepted"
        candidate.resolved_at = datetime.now(UTC)
        _append_memory_event(
            session,
            memory,
            "confirmed",
            "conversation_confirmed",
            candidate.conversation_id,
            candidate.source_message_id,
            candidate.id,
        )

    @staticmethod
    def dismiss_candidate(candidate: UserMemoryCandidate) -> None:
        """标记为已忽略，避免在同一会话中重复展示。"""
        candidate.status = "dismissed"
        candidate.resolved_at = datetime.now(UTC)


def extract_memory_candidate(question: str) -> MemoryCandidateProposal | None:
    """只从稳定背景与偏好表达生成候选，不把临时业务问题写成长期记忆。"""
    normalized = question.strip()
    if _REMEMBER_PREFIX.match(normalized) or _FORGET_PREFIX.match(normalized):
        return None
    for pattern in _WORK_PROFILE_PATTERNS:
        matched = pattern.match(normalized)
        if matched is not None:
            content = _normalize_memory_content(f"负责{matched.group(1)}")
            return _candidate_if_allowed("work_profile", content)
    for pattern in _FOCUS_DIRECTION_PATTERNS:
        matched = pattern.match(normalized)
        if matched is not None:
            content = _normalize_memory_content(f"重点关注{matched.group(1)}")
            return _candidate_if_allowed("focus_direction", content)
    for pattern in _CANDIDATE_PATTERNS:
        matched = pattern.match(normalized)
        if matched is None:
            continue
        content = _normalize_memory_content(matched.group(1))
        return _candidate_if_allowed(_classify_memory_type(content), content)
    return None


def _candidate_if_allowed(memory_type: str, content: str) -> MemoryCandidateProposal | None:
    if len(content) < 2 or _contains_forbidden_content(content) or _contains_sensitive_content(content):
        return None
    return MemoryCandidateProposal(memory_type, content)


def _parse_remember_content(raw_content: str) -> _RememberContent:
    """支持“记住到 2026-12-31：...”这一显式有效期写法。"""
    content = _normalize_memory_content(raw_content)
    matched = _EXPIRES_ON_PREFIX.match(content)
    if matched is None:
        return _RememberContent(content)
    try:
        expires_on = datetime.strptime(matched.group(1), "%Y-%m-%d").date()
    except ValueError:
        return _RememberContent("", error_message="有效期请使用 YYYY-MM-DD，例如“记住到 2026-12-31：重点关注直播转化”。")
    expires_at = datetime.combine(expires_on, time.max, tzinfo=UTC)
    if expires_at <= datetime.now(UTC):
        return _RememberContent("", error_message="长期记忆的有效期必须晚于当前时间。")
    return _RememberContent(_normalize_memory_content(matched.group(2)), expires_at=expires_at)


def _default_expiry(memory_type: str) -> datetime | None:
    """稳定工作背景不过期；近期关注方向默认在配置天数后自动失效。"""
    if memory_type != "focus_direction":
        return None
    return datetime.now(UTC) + timedelta(days=get_settings().user_memory_focus_direction_ttl_days)


async def _replace_memory_type(
    session: AsyncSession,
    user_id: str,
    memory_type: str,
    content: str,
    expires_at: datetime | None,
    conversation_id: str | None,
    source_message_id: str | None,
) -> MemoryCommandResult:
    """显式更新会替代同类型有效记忆，避免新旧工作背景或关注方向同时生效。"""
    if len(content) < 2:
        return MemoryCommandResult("请在“更新…”后补充至少两个字符的记忆内容。")
    current = list(
        await session.scalars(
            select(UserMemory).where(UserMemory.user_id == user_id, UserMemory.memory_type == memory_type, UserMemory.status == "active")
        )
    )
    for record in current:
        record.status = "superseded"
        _append_memory_event(session, record, "superseded", "user_update", conversation_id, source_message_id)
    memory = UserMemory(
        user_id=user_id,
        memory_type=memory_type,
        content=content,
        source="user_update",
        status="active",
        expires_at=expires_at,
    )
    session.add(memory)
    await session.flush()
    _append_memory_event(
        session,
        memory,
        "created",
        "user_update",
        conversation_id,
        source_message_id,
        details={"replaced_memory_ids": [record.id for record in current]},
    )
    await session.commit()
    expires_display = f"（有效至 {expires_at.date().isoformat()}）" if expires_at else ""
    return MemoryCommandResult(f"已更新：{content}{expires_display}")


async def _expire_active_memories(session: AsyncSession, user_id: str, now: datetime) -> int:
    """读取或写入前惰性标记过期记录，避免到期内容继续被计数或采用。"""
    records = list(
        await session.scalars(
            select(UserMemory).where(
                UserMemory.user_id == user_id,
                UserMemory.status == "active",
                UserMemory.expires_at.is_not(None),
                UserMemory.expires_at <= now,
            )
        )
    )
    for record in records:
        record.status = "expired"
        _append_memory_event(session, record, "expired", "system_expiry")
    return len(records)


async def _active_memory_count(session: AsyncSession, user_id: str) -> int:
    return int(
        await session.scalar(
            select(func.count(UserMemory.id)).where(UserMemory.user_id == user_id, UserMemory.status == "active")
        )
        or 0
    )


def _append_memory_event(
    session: AsyncSession,
    memory: UserMemory,
    event_type: str,
    source: str,
    conversation_id: str | None = None,
    message_id: str | None = None,
    candidate_id: str | None = None,
    details: dict[str, object] | None = None,
) -> None:
    """只记录记忆 ID、生命周期事件和来源定位，不把正文重复写入审计表。"""
    session.add(
        UserMemoryEvent(
            user_id=memory.user_id,
            memory_id=memory.id,
            event_type=event_type,
            source=source,
            candidate_id=candidate_id,
            conversation_id=conversation_id,
            message_id=message_id,
            details=details or {},
        )
    )


def _normalize_memory_content(content: str) -> str:
    """压缩命令分隔符和无意义空白，使展示与去重结果稳定。"""
    return content.strip(" ：:，,。！!？?\t\n")


def _contains_forbidden_content(content: str) -> bool:
    normalized = content.lower()
    return any(keyword in normalized for keyword in _FORBIDDEN_MEMORY_KEYWORDS)


def _contains_sensitive_content(content: str) -> bool:
    normalized = content.lower()
    return any(keyword in normalized for keyword in _SENSITIVE_KEYWORDS) or any(
        pattern.search(content) is not None for pattern in _SENSITIVE_PATTERNS
    )


def _classify_memory_type(content: str) -> str:
    """根据少量可解释关键词确定展示和召回优先级，不调用额外模型。"""
    if any(keyword in content for keyword in ("负责", "岗位", "业务线", "主营")):
        return "work_profile"
    if any(keyword in content for keyword in ("近期", "当前", "目前", "重点关注")):
        return "focus_direction"
    if any(keyword in content for keyword in ("复盘", "分析", "归因", "维度", "渠道", "品类", "拆分")):
        return "analysis_preference"
    if any(keyword in content for keyword in ("回答", "回复", "结论", "简洁", "篇幅", "格式", "语气")):
        return "answer_preference"
    return "focus_topic"


def _select_relevant_memories(records: list[UserMemory], question: str) -> list[UserMemory]:
    """以类型和明确话题进行轻量排序，不为少量用户偏好引入向量检索。"""
    normalized_question = question.lower()
    is_analysis_question = any(keyword in question for keyword in _ANALYSIS_KEYWORDS)

    def priority(record: UserMemory) -> tuple[int, int, float]:
        if record.memory_type == "answer_preference":
            type_priority = 0
        elif record.memory_type == "analysis_preference":
            type_priority = 1 if is_analysis_question else 3
        elif record.memory_type == "work_profile":
            type_priority = 1 if is_analysis_question else 2
        elif record.memory_type == "focus_direction":
            type_priority = 2
        elif record.memory_type == "focus_topic":
            type_priority = 1 if record.content.lower() in normalized_question else 2
        else:
            type_priority = 4
        return (type_priority, -record.use_count, -record.updated_at.timestamp())

    return sorted(records, key=priority)[:MAX_MEMORIES_PER_QUERY]


def _selection_reason(record: UserMemory, question: str) -> str:
    """为运行审计提供不含正文的可解释采用原因。"""
    if record.memory_type == "answer_preference":
        return "回答呈现偏好"
    if record.memory_type == "analysis_preference":
        return "分析类问题匹配"
    if record.memory_type == "work_profile":
        return "稳定工作背景"
    if record.memory_type == "focus_direction":
        return "未过期近期关注方向"
    return "关注主题与当前问题匹配" if record.content.lower() in question.lower() else "有效关注主题"
