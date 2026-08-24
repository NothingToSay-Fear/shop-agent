"""将完整回答拆分为适合 SSE 推送的文本片段。"""


def split_answer_fragments(text: str, size: int = 24) -> list[str]:
    """保持短片段可读，同时避免一次性推送完整回答。"""
    return [text[index : index + size] for index in range(0, len(text), size)]
