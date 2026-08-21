from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import conversations, feedback, metrics, tasks
from app.config import get_settings
from app.database import close_database, create_tables

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Docker 会在 Uvicorn 前执行 Alembic；此处也让全新的本地数据库可直接使用。
    await create_tables()
    yield
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
app.include_router(tasks.router)
app.include_router(feedback.router)
app.include_router(metrics.router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """供 Docker 健康检查使用的轻量、无依赖接口。"""
    return {"status": "ok"}
