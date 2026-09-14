import pytest

from app.agent.operation_agent import OperationAgent
from app.agent.tools import AgentToolTracker
from app.agent.workflow import AgentWorkflow, WorkflowResult
from app.config import Settings


@pytest.mark.asyncio
async def test_agent_stream_events_forwards_workflow_status_and_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSE 适配层只转发正式工作流产出的进度与回答，不提供绕过检索的外部上下文入口。"""

    async def fake_answer(
        _: AgentWorkflow,
        user_input: str,
        __: str,
        on_status,
        *___: object,
    ) -> WorkflowResult:
        assert user_input == "查询本周 GMV"
        await on_status("routing", "正在判断问题类型…")
        return WorkflowResult("这是受控工作流的回答。", "指标定义：paid_gmv", AgentToolTracker())

    monkeypatch.setattr(AgentWorkflow, "answer", fake_answer)
    agent = OperationAgent(Settings(llm_api_key=None, llm_model=None))

    events = [event async for event in agent.stream_events("查询本周 GMV")]

    assert events[0].event_type == "status"
    assert events[0].phase == "routing"
    assert "".join(event.content for event in events if event.event_type == "chunk") == "这是受控工作流的回答。"
    assert agent.data_references == "指标定义：paid_gmv"
