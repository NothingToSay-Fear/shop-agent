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
