"""本地嵌入模型的加载与向量化，不依赖外部 Embeddings API。"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from fastapi.concurrency import run_in_threadpool

from app.config import Settings

logger = logging.getLogger(__name__)


@lru_cache
def _load_model(model_path: str, device: str):
    """按路径和设备缓存模型实例，确保一个 API 进程只加载一次。"""
    from sentence_transformers import SentenceTransformer

    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"本地嵌入模型目录不存在：{path}")
    # local_files_only 防止配置错误时隐式访问网络下载模型。
    return SentenceTransformer(str(path), device=device, local_files_only=True)


def _encode(model_path: str, device: str, texts: tuple[str, ...]) -> list[list[float]]:
    """在线程池中执行同步的 CPU/GPU 推理，避免阻塞 FastAPI 事件循环。"""
    model = _load_model(model_path, device)
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


async def embed_texts(texts: list[str], settings: Settings) -> list[list[float]] | None:
    """用已挂载的模型目录生成归一化向量；模型不可用时返回空结果。"""
    if not texts or not settings.local_embedding_enabled:
        return None
    try:
        return await run_in_threadpool(
            _encode,
            settings.local_embedding_model_path or "",
            settings.local_embedding_device,
            tuple(texts),
        )
    except Exception as error:
        logger.warning("本地嵌入模型不可用，无法执行 RAG 语义检索：%s", error)
        return None


async def preload_model(settings: Settings) -> bool:
    """在应用启动期预加载模型，提前暴露路径或依赖问题。"""
    if not settings.local_embedding_enabled:
        logger.warning("未配置本地嵌入模型目录，RAG 语义检索不可用。")
        return False
    vectors = await embed_texts(["本地嵌入模型预热"], settings)
    if vectors is None:
        return False
    logger.info("本地嵌入模型已加载：%s", settings.local_embedding_model_id)
    return True
