from contextlib import asynccontextmanager
import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import conversations, knowledge, metrics
from app.config import get_settings
from app.database import SessionLocal, close_database, create_tables
from app.services.local_embeddings import preload_model
from app.services.agent_audit import cleanup_expired_agent_audits

settings = get_settings()
# 复用 Uvicorn 已配置的标准输出处理器，确保 Docker 日志可看到清理任务状态。
logger = logging.getLogger("uvicorn.error")


async def _audit_cleanup_loop() -> None:
    """启动时及随后每 24 小时清理一次过期运行审计。"""
    while True:
        try:
            async with SessionLocal() as session:
                await cleanup_expired_agent_audits(session)
        except Exception:
            # 清理失败不能影响对话服务；异常正文由服务日志记录，且不含用户输入。
            logger.exception("agent_audit_cleanup_failed")
        await asyncio.sleep(24 * 60 * 60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Docker 会在 Uvicorn 前执行 Alembic；此处也让全新的本地数据库可直接使用。
    await create_tables()
    # 模型目录已挂载时仅在启动期加载一次；缺失时记录日志，指标 RAG 不提供数据上下文。
    await preload_model(settings)
    cleanup_task = asyncio.create_task(_audit_cleanup_loop(), name="agent-audit-cleanup")
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        await close_database()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
# Web 客户端运行在独立容器/进程中，因此浏览器请求需要启用 CORS。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(conversations.router)
app.include_router(knowledge.router)
app.include_router(metrics.router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """供 Docker 健康检查使用的轻量、无依赖接口。"""
    return {"status": "ok"}
