"""Docker 会话摘要 Worker 入口。"""

import asyncio
import logging

from app.services.conversations.summary import run_summary_worker


if __name__ == "__main__":
    # 独立 Worker 没有 Uvicorn 的日志初始化，主动输出任务状态便于 Docker 排查。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(run_summary_worker())
