"""知识库候选召回所需的稳定中文分词文本构造。"""

from __future__ import annotations

from app.services.hybrid_retrieval import tokenize_for_bm25


def build_knowledge_search_terms(
    title: str, heading: str | None, content: str
) -> str:
    """将标题、层级与正文转为可被 PostgreSQL 全文索引匹配的 jieba 词项。"""
    source = "\n".join(part for part in (title, heading, content) if part)
    return " ".join(tokenize_for_bm25(source))
