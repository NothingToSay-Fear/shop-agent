import pytest
from langchain_core.messages import AIMessageChunk

from app.agent.answer_generator import AnswerGenerator
from app.config import Settings


def test_evidence_response_only_returns_verified_context_without_llm() -> None:
    """无 LLM 时不能根据关键词编造诊断，只能返回同轮受控证据。"""
    response = AnswerGenerator(Settings()).generate_evidence_response(
        "【经营指标】\n支付 GMV：1,000 元。"
    )

    assert "支付 GMV：1,000 元。" in response
    assert "初步诊断" not in response
    assert "演示模式" not in response


def test_evidence_response_explains_when_no_verified_context_exists() -> None:
    """无 LLM 且没有检索依据时，系统明确拒绝生成无依据回答。"""
    response = AnswerGenerator(Settings()).generate_evidence_response(None, model_error=True)

    assert "模型服务暂时不可用" in response
    assert "无法生成回答" in response


@pytest.mark.asyncio
async def test_stream_with_llm_forwards_only_ai_text_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeAgent:
        async def astream(self, *_args: object, **_kwargs: object):
            yield (AIMessageChunk(content="第一段"), {"langgraph_node": "model"})
            yield (AIMessageChunk(content=[{"type": "text", "text": "第二段"}]), {"langgraph_node": "model"})
            yield (object(), {"langgraph_node": "tools"})

    generator = AnswerGenerator(Settings(llm_api_key="test", llm_model="test-model"))
    monkeypatch.setattr(generator, "_build_deep_agent", lambda _tools: _FakeAgent())

    chunks = [
        chunk
        async for chunk in generator.stream_with_llm("测试问题", "受控证据", [])
    ]

    assert chunks == ["第一段", "第二段"]
