from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

# 每个进程复用一个引擎和会话工厂；请求处理函数获取短生命周期会话。
settings = get_settings()
engine = create_async_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """为 FastAPI 依赖项提供支持事务的数据库会话。"""
    async with SessionLocal() as session:
        yield session


async def create_tables() -> None:
    """在本地尚未执行 Alembic 迁移时创建数据表。"""
    from app.models import Base

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def close_database() -> None:
    """应用关闭时释放连接池资源。"""
    await engine.dispose()
