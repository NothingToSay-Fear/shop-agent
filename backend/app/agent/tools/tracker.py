"""汇总一次 Agent 运行中各检索工具的返回依据。"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.knowledge_rag import KnowledgeQueryContext
from app.services.metric_rag import MetricQueryContext
from app.services.web_search import WebSearchContext


@dataclass
class AgentToolTracker:
    """收集实际调用的工具依据，供演示回答与会话消息持久化使用。"""

    metric_context: MetricQueryContext | None = None
    knowledge_context: KnowledgeQueryContext | None = None
    knowledge_miss: bool = False
    web_context: WebSearchContext | None = None
    web_search_miss: bool = False

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
