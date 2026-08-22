import pytest

from app.config import Settings
from app.services.local_embeddings import embed_texts, preload_model
from app.services.metric_rag import _query_embedding


@pytest.mark.asyncio
async def test_local_embedding_requires_model_path() -> None:
    """未挂载模型时不应请求网络，也不应使用替代向量算法。"""
    settings = Settings(local_embedding_model_path=None)

    assert await embed_texts(["支付转化率"], settings) is None
    assert await preload_model(settings) is False
    vector = await _query_embedding("支付转化率", settings)

    assert vector is None
