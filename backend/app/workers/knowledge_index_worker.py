"""Docker Worker 的知识库索引任务入口。"""

import asyncio

from app.services.knowledge.indexer import run_worker


if __name__ == "__main__":
    asyncio.run(run_worker())
