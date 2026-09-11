"""Shop Agent 的受控工具集合。"""

from langchain_core.tools import BaseTool

from app.agent.tools.knowledge_rag import build_knowledge_rag_tool
from app.agent.tools.metric_rag import build_metric_rag_tool
from app.agent.tools.tracker import AgentToolTracker
from app.agent.tools.web_search import build_web_search_tool
from app.config import Settings


def build_agent_tools(
    tracker: AgentToolTracker, settings: Settings, user_id: str | None = None
) -> list[BaseTool]:
    """按职责组合指标、知识库与联网搜索工具。"""
    return [
        build_metric_rag_tool(tracker, settings),
        build_knowledge_rag_tool(tracker, settings, user_id),
        build_web_search_tool(tracker, settings),
    ]


__all__ = ["AgentToolTracker", "build_agent_tools"]
