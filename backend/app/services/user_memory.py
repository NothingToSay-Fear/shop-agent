"""按用户召回、确认和维护跨会话长期偏好。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re

from sqlalchemy import delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import UserMemory, UserMemoryCandidate

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


@dataclass(frozen=True)
class MemoryItem:
    """传给 Agent 的最小化长期记忆，不携带内部状态或用户标识。"""

    id: str
    memory_type: str
    content: str


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
        }
        return "\n".join(
            f"- {labels.get(item.memory_type, '用户偏好')}：{item.content}" for item in self.items
        )

    @property
    def audit_summary(self) -> str | None:
        if not self.items:
            return None
        return f"已采用 {len(self.items)} 条用户长期记忆"


@dataclass(frozen=True)
class MemoryCommandResult:
    """显式记忆命令的确定性响应，不触发 Agent、RAG 或业务查询。"""

    message: str


@dataclass(frozen=True)
class MemoryCandidateProposal:
    """从自然表达提取、但尚未确认的偏好候选。"""

    memory_type: str
    content: str


class MemoryService:
    """将可确认的偏好压缩为每轮最多三条的 Agent 上下文。"""

    @staticmethod
    async def retrieve_for_query(
        session: AsyncSession, user_id: str, question: str
    ) -> UserMemoryContext:
        """只读取当前用户有效记忆，并按问题场景选择少量偏好。"""
        now = datetime.now(UTC)
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
        return UserMemoryContext(
            tuple(MemoryItem(item.id, item.memory_type, item.content) for item in selected)
        )

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
        session: AsyncSession, user_id: str, question: str
    ) -> MemoryCommandResult | None:
        """识别“记住、忘记、清除所有记忆”等显式命令并在用户域内执行。"""
        normalized = question.strip()
        if normalized in {"清除所有记忆", "清空所有记忆", "忘记所有记忆"}:
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

        remember_match = _REMEMBER_PREFIX.match(normalized)
        if remember_match:
            content = _normalize_memory_content(remember_match.group(1))
            if _contains_forbidden_content(content):
                return MemoryCommandResult("长期记忆只保存回答偏好、分析习惯或关注主题，不能保存指标口径或业务数据。")
            if len(content) < 2:
                return MemoryCommandResult("请在“记住”后补充至少两个字符的偏好内容。")
            memory_type = _classify_memory_type(content)
            existing = await session.scalar(
                select(UserMemory).where(
                    UserMemory.user_id == user_id,
                    UserMemory.memory_type == memory_type,
                    UserMemory.content == content,
                )
            )
            if existing is None:
                session.add(
                    UserMemory(
                        user_id=user_id,
                        memory_type=memory_type,
                        content=content,
                        source="user_explicit",
                        status="active",
                    )
                )
            else:
                existing.status = "active"
                existing.source = "user_explicit"
            await session.commit()
            return MemoryCommandResult(f"已记住：{content}")

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
        existing_memory = await session.scalar(
            select(UserMemory.id).where(
                UserMemory.user_id == user_id,
                UserMemory.memory_type == proposal.memory_type,
                UserMemory.content == proposal.content,
                UserMemory.status == "active",
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
        )
        session.add(candidate)
        await session.commit()
        await session.refresh(candidate)
        return candidate

    @staticmethod
    async def accept_candidate(session: AsyncSession, candidate: UserMemoryCandidate) -> None:
        """把已确认候选转化为用户记忆；同内容重复确认时复用已有记录。"""
        memory = await session.scalar(
            select(UserMemory).where(
                UserMemory.user_id == candidate.user_id,
                UserMemory.memory_type == candidate.memory_type,
                UserMemory.content == candidate.content,
            )
        )
        if memory is None:
            session.add(
                UserMemory(
                    user_id=candidate.user_id,
                    memory_type=candidate.memory_type,
                    content=candidate.content,
                    source="conversation_confirmed",
                    confidence=candidate.confidence,
                    status="active",
                )
            )
        else:
            memory.status = "active"
            memory.source = "conversation_confirmed"
        candidate.status = "accepted"
        candidate.resolved_at = datetime.now(UTC)

    @staticmethod
    def dismiss_candidate(candidate: UserMemoryCandidate) -> None:
        """标记为已忽略，避免在同一会话中重复展示。"""
        candidate.status = "dismissed"
        candidate.resolved_at = datetime.now(UTC)


def extract_memory_candidate(question: str) -> MemoryCandidateProposal | None:
    """从有限的偏好句式中提取候选，避免把普通业务问题或指标条件误存为记忆。"""
    normalized = question.strip()
    if _REMEMBER_PREFIX.match(normalized) or _FORGET_PREFIX.match(normalized):
        return None
    for pattern in _CANDIDATE_PATTERNS:
        matched = pattern.match(normalized)
        if matched is None:
            continue
        content = _normalize_memory_content(matched.group(1))
        if len(content) < 2 or _contains_forbidden_content(content):
            return None
        return MemoryCandidateProposal(_classify_memory_type(content), content)
    return None


def _normalize_memory_content(content: str) -> str:
    """压缩命令分隔符和无意义空白，使展示与去重结果稳定。"""
    return content.strip(" ：:，,。！!？?\t\n")


def _contains_forbidden_content(content: str) -> bool:
    normalized = content.lower()
    return any(keyword in normalized for keyword in _FORBIDDEN_MEMORY_KEYWORDS)


def _classify_memory_type(content: str) -> str:
    """根据少量可解释关键词确定展示和召回优先级，不调用额外模型。"""
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
        elif record.memory_type == "focus_topic":
            type_priority = 1 if record.content.lower() in normalized_question else 2
        else:
            type_priority = 4
        return (type_priority, -record.use_count, -record.updated_at.timestamp())

    return sorted(records, key=priority)[:MAX_MEMORIES_PER_QUERY]
