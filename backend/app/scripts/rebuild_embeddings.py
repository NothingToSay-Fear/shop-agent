"""使用当前本地嵌入模型重建指标与知识库向量。"""

from __future__ import annotations

import asyncio

from sqlalchemy import select

from app.config import get_settings
from app.database import SessionLocal
from app.models import MetricDefinition
from app.services.models.embeddings import embed_texts
from app.services.knowledge.retrieval import reindex_knowledge_chunks


def _retrieval_text(definition: MetricDefinition) -> str:
    """保持与指标 RAG 一致的检索文本构成，避免索引和查询口径不一致。"""
    return " ".join(
        [definition.name, definition.description, *definition.aliases, definition.metric_code]
    )


async def reindex_embeddings() -> None:
    """批量覆盖指标定义和知识库片段向量，并记录当前模型标识。"""
    settings = get_settings()
    async with SessionLocal() as session:
        definitions = list(
            await session.scalars(select(MetricDefinition).where(MetricDefinition.enabled.is_(True)))
        )
        vectors = await embed_texts([_retrieval_text(item) for item in definitions], settings)
        if vectors is None:
            raise RuntimeError("本地嵌入模型不可用，无法重建向量索引")
        for definition, vector in zip(definitions, vectors, strict=True):
            definition.embedding = vector
            definition.embedding_model = settings.local_embedding_model_id
        await session.commit()
        knowledge_count = await reindex_knowledge_chunks(session)
        print(
            f"已使用 {settings.local_embedding_model_id} 重建 {len(definitions)} 个指标向量"
            f"和 {knowledge_count} 个知识库片段向量。"
        )


if __name__ == "__main__":
    asyncio.run(reindex_embeddings())
