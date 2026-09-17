import asyncio

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

    allow_result = asyncio.Event()
    result_returned = asyncio.Event()

    async def fake_answer(
        _: AgentWorkflow,
        user_input: str,
        __: str,
        on_status,
        *args: object,
    ) -> WorkflowResult:
        assert user_input == "查询本周 GMV"
        await on_status("routing", "正在判断问题类型…")
        on_chunk = args[-1]
        assert callable(on_chunk)
        await on_chunk("这是实时输出的")
        await on_chunk("受控工作流回答。")
        await allow_result.wait()
        result_returned.set()
        return WorkflowResult("这是受控工作流的回答。", "指标定义：paid_gmv", AgentToolTracker())

    monkeypatch.setattr(AgentWorkflow, "answer", fake_answer)
    agent = OperationAgent(Settings(llm_api_key=None, llm_model=None))

    stream = agent.stream_events("查询本周 GMV")
    first_event = await anext(stream)
    first_chunk = await anext(stream)
    second_chunk = await anext(stream)

    assert first_event.event_type == "status"
    assert first_event.phase == "routing"
    assert [first_chunk.content, second_chunk.content] == ["这是实时输出的", "受控工作流回答。"]
    assert not result_returned.is_set()

    allow_result.set()
    assert [event async for event in stream] == []
    assert agent.data_references == "指标定义：paid_gmv"
