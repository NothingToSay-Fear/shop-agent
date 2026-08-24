from app.agent.review_agent import (
    GENERAL_PURPOSE_AGENT_NAME,
    REVIEW_AGENT_NAME,
    build_general_subagent,
    build_review_subagent,
)
from app.agent.tools import AgentToolTracker, build_agent_tools
from app.config import Settings


def test_review_subagent_only_receives_declared_tools() -> None:
    """复盘子 Agent 的职责必须收敛到已声明的受控工具。"""
    tools = [object(), object()]

    subagent = build_review_subagent(tools)

    assert subagent["name"] == REVIEW_AGENT_NAME
    assert subagent["tools"] == tools
    assert "复盘" in str(subagent["description"])


def test_general_subagent_overrides_framework_default_permissions() -> None:
    """显式通用子 Agent 应覆盖 DeepAgent 自动添加的同名 Agent。"""
    subagent = build_general_subagent([])

    assert subagent["name"] == GENERAL_PURPOSE_AGENT_NAME
    assert "不执行文件" in str(subagent["description"])


def test_agent_tool_tracker_only_persists_actual_references() -> None:
    """未调用任何工具时，不应把虚假的指标或文档来源写入会话。"""
    tracker = AgentToolTracker()

    assert tracker.data_context is None
    assert tracker.references == "演示模式：尚未检索到相关数据、资料或公开网页来源"


def test_main_agent_tools_include_controlled_web_search() -> None:
    """联网能力必须以受控工具形式加入，而不是向模型暴露 HTTP 客户端。"""
    tools = build_agent_tools(AgentToolTracker(), Settings())

    assert [tool.name for tool in tools] == [
        "query_metric_rag",
        "query_knowledge_rag",
        "search_web",
    ]
