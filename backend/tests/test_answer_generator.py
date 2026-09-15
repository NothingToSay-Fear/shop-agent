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
