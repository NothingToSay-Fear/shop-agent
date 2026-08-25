"""汇总一次 Agent 运行中各检索工具的返回依据。"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from app.services.knowledge_rag import KnowledgeQueryContext
from app.services.metric_rag import MetricQueryContext
from app.services.intent_router import RetrievalRoute
from app.services.web_search import WebSearchContext


@dataclass(frozen=True)
class ToolCallAudit:
    """一条仅含摘要和引用 ID 的工具调用轨迹。"""

    tool_name: str
    input_summary: str
    result_summary: str
    reference_ids: tuple[str, ...]
    status: str
    duration_ms: int
    error_code: str | None = None


@dataclass
class AgentToolTracker:
    """收集实际工具依据和审计摘要，不保存原始提问或工具完整输出。"""

    metric_context: MetricQueryContext | None = None
    knowledge_context: KnowledgeQueryContext | None = None
    knowledge_miss: bool = False
    web_context: WebSearchContext | None = None
    web_search_miss: bool = False
    route: RetrievalRoute | None = None
    tool_calls: list[ToolCallAudit] = field(default_factory=list)
    tool_invocation_counts: dict[str, int] = field(default_factory=dict)

    def set_route(self, route: RetrievalRoute) -> None:
        """记录最终路由，供运行审计关联本次实际编排决策。"""
        self.route = route

    def start_tool_call(self) -> float:
        """为工具包装器提供单调时钟起点。"""
        return perf_counter()

    def reserve_tool_call(self, tool_name: str, max_calls_per_run: int) -> bool:
        """为一次真实工具执行预留额度，避免模型在计划完成后重复检索。"""
        current_count = self.tool_invocation_counts.get(tool_name, 0)
        if current_count >= max_calls_per_run:
            return False
        self.tool_invocation_counts[tool_name] = current_count + 1
        return True

    def record_tool_call(
        self,
        *,
        tool_name: str,
        input_summary: str,
        result_summary: str,
        reference_ids: tuple[str, ...] = (),
        status: str,
        started_at: float,
        error_code: str | None = None,
    ) -> None:
        """追加一条脱敏工具轨迹；该方法同时覆盖模型自行调用工具的场景。"""
        self.tool_calls.append(
            ToolCallAudit(
                tool_name=tool_name,
                input_summary=input_summary,
                result_summary=result_summary,
                reference_ids=reference_ids,
                status=status,
                duration_ms=max(0, round((perf_counter() - started_at) * 1000)),
                error_code=error_code,
            )
        )

    @property
    def reference_ids(self) -> list[str]:
        """去重汇总实际命中的指标、片段和网页链接标识。"""
        identifiers: list[str] = []
        for call in self.tool_calls:
            for reference_id in call.reference_ids:
                if reference_id not in identifiers:
                    identifiers.append(reference_id)
        return identifiers

    @property
    def data_context(self) -> str | None:
        """将各工具结果组装成带来源类型的受限上下文。"""
        parts: list[str] = []
        if self.metric_context:
            parts.append(f"【经营指标】\n{self.metric_context.text}")
        if self.knowledge_context:
            parts.append(f"【知识库资料】\n{self.knowledge_context.text}")
        if self.knowledge_miss and not self.knowledge_context:
            parts.append("【知识库资料】\n知识库中未检索到足以回答该问题的资料，请明确说明资料不足。")
        if self.web_context:
            parts.append(self.web_context.text)
        if self.web_search_miss and not self.web_context:
            parts.append("【联网公开资料】\n未检索到可用公开资料，或联网搜索尚未配置/暂不可用。")
        return "\n\n".join(parts) or None

    @property
    def references(self) -> str:
        """返回实际检索到的来源，避免把未调用的能力写入会话记录。"""
        references: list[str] = []
        if self.metric_context:
            references.append(f"内置模拟经营数据：{', '.join(self.metric_context.metric_codes)}")
        if self.knowledge_context:
            references.append(self.knowledge_context.references)
        if self.web_context:
            references.append(f"联网公开资料：{self.web_context.references}")
        if references:
            return "；".join(references)
        misses: list[str] = []
        if self.knowledge_miss:
            misses.append("知识库未检索到相关资料")
        if self.web_search_miss:
            misses.append("联网搜索未检索到相关资料或尚未配置")
        if misses:
            return "；".join(misses)
        return "演示模式：尚未检索到相关数据、资料或公开网页来源"
