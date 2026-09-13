"""知识库候选召回所需的稳定中文分词文本构造。"""

from __future__ import annotations

from app.services.hybrid_retrieval import tokenize_for_bm25


CONTENT_TYPE_LABELS = {
    "paragraph": "正文",
    "list": "列表",
    "table": "表格",
    "mixed": "混合内容",
}


def build_knowledge_retrieval_text(
    title: str,
    heading_path: str | None,
    content_type: str,
    content: str,
) -> str:
    """统一向量、词面与精排使用的资料语义上下文。"""
    parts = [title]
    if heading_path:
        parts.append(heading_path)
    if content_type != "paragraph":
        parts.append(CONTENT_TYPE_LABELS.get(content_type, content_type))
    parts.append(content)
    return "\n".join(part for part in parts if part.strip())


def build_knowledge_search_terms(
    title: str, heading_path: str | None, content_type: str, content: str
) -> str:
    """将标题、层级与正文转为可被 PostgreSQL 全文索引匹配的 jieba 词项。"""
    source = build_knowledge_retrieval_text(title, heading_path, content_type, content)
    return " ".join(tokenize_for_bm25(source))
