"""本地 CrossEncoder 精排；模型未准备好时由 RRF 顺序安全降级。"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from fastapi.concurrency import run_in_threadpool

from app.config import Settings

logger = logging.getLogger(__name__)


@lru_cache
def _load_reranker(model_path: str, device: str):
    """按路径和设备缓存 CrossEncoder，禁止推理期间联网下载模型。"""
    from sentence_transformers import CrossEncoder

    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"本地精排模型目录不存在：{path}")
    return CrossEncoder(str(path), device=device, local_files_only=True)


def _score_pairs(model_path: str, device: str, question: str, texts: tuple[str, ...]) -> list[float]:
    """在线程池中执行同步精排推理，避免阻塞 API 事件循环。"""
    model = _load_reranker(model_path, device)
    scores = model.predict([(question, text) for text in texts], show_progress_bar=False)
    return [float(score) for score in scores]


async def rerank_texts(
    question: str, texts: list[str], settings: Settings
) -> list[float] | None:
    """返回与 texts 对齐的相关度；模型缺失时返回 None 交由调用方使用 RRF 结果。"""
    if not texts or not settings.local_reranker_enabled:
        return None
    try:
        return await run_in_threadpool(
            _score_pairs,
            settings.local_reranker_model_path or "",
            settings.local_reranker_device,
            question,
            tuple(texts),
        )
    except Exception as error:
        logger.warning("本地精排模型不可用，已保留 RRF 融合顺序：%s", error)
        return None


async def preload_reranker_model(settings: Settings) -> bool:
    """启动期预热精排模型，尽早暴露模型目录或依赖问题。"""
    if not settings.local_reranker_enabled:
        return False
    scores = await rerank_texts("本地精排预热", ["本地精排预热"], settings)
    if scores is None:
        return False
    logger.info("本地精排模型已加载：%s", settings.local_reranker_model_id)
    return True
