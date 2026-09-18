from app.agent.tools import AgentToolTracker, build_agent_tools
from app.config import Settings


def test_agent_tool_tracker_only_persists_actual_references() -> None:
    """未调用任何工具时，不应把虚假的指标或文档来源写入会话。"""
    tracker = AgentToolTracker()

    assert tracker.data_context is None
    assert tracker.references == "本轮尚未检索到相关数据、资料或公开网页来源"


def test_main_agent_tools_include_controlled_web_search() -> None:
    """联网能力必须以受控工具形式加入，而不是向模型暴露 HTTP 客户端。"""
    tools = build_agent_tools(AgentToolTracker(), Settings())

    assert [tool.name for tool in tools] == [
        "query_metric_rag",
        "query_knowledge_rag",
        "search_web",
    ]
