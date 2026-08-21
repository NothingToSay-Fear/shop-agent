from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.services.metrics import get_metrics_overview

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/overview")
async def get_overview(session: AsyncSession = Depends(get_session)) -> dict[str, object]:
    """返回内置模拟经营数据的最近两周汇总与环比结果。"""
    overview = await get_metrics_overview(session)
    if overview is None:
        return {"source": "demo", "data_available": False}
    return {"source": "demo", "data_available": True, **overview.as_dict()}
