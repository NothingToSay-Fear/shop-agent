import pytest

from app.agent.operation_agent import OperationAgent
from app.config import Settings


@pytest.mark.asyncio
async def test_demo_agent_returns_operational_guidance() -> None:
    agent = OperationAgent(Settings(llm_api_key=None, llm_model=None))
    context = "数据来源：内置模拟经营数据。GMV 12,000.00 元，GMV 环比 -8.00%。"
    answer = "".join(
        [chunk async for chunk in agent.stream("分析本周 GMV 环比下降", context)]
    )

    assert "经营数据分析" in answer
    assert "12,000.00" in answer
    assert "验证指标" in answer


@pytest.mark.asyncio
async def test_demo_agent_does_not_invent_when_knowledge_is_missing() -> None:
    """知识库检索未命中时，离线回答也必须明确资料不足。"""
    agent = OperationAgent(Settings(llm_api_key=None, llm_model=None))
    answer = "".join(
        [
            chunk
            async for chunk in agent.stream(
                "活动报名条件是什么？",
                "【知识库资料】\n知识库中未检索到足以回答该问题的资料，请明确说明资料不足。",
            )
        ]
    )

    assert "未检索到" in answer


@pytest.mark.asyncio
async def test_demo_agent_marks_hybrid_context_as_combined_evidence() -> None:
    """同一问题同时命中数据和资料时，应保留两类依据而非互相覆盖。"""
    agent = OperationAgent(Settings(llm_api_key=None, llm_model=None))
    context = "【经营指标】\nGMV 环比 -8%。\n\n【知识库资料】\n历史活动曾使用定向券。"
    answer = "".join([chunk async for chunk in agent.stream("GMV 下滑怎么办？", context)])

    assert "综合依据" in answer
    assert "GMV 环比" in answer
    assert "定向券" in answer


@pytest.mark.asyncio
async def test_agent_stream_events_expose_progress_before_answer_chunks() -> None:
    """前端应能在回答文本前收到不含原文的执行阶段提示。"""
    agent = OperationAgent(Settings(llm_api_key=None, llm_model=None))

    events = [
        event
        async for event in agent.stream_events(
            "分析本周 GMV 环比下降",
            "数据来源：内置模拟经营数据。GMV 12,000.00 元，GMV 环比 -8.00%。",
        )
    ]

    assert events[0].event_type == "status"
    assert events[0].phase == "generation"
    assert any(event.event_type == "chunk" for event in events)
