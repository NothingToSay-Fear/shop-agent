"""将开源嵌入模型下载到宿主机挂载的本地模型目录。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODEL_ID = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL_ID = "BAAI/bge-reranker-base"


def main() -> None:
    """下载模型推理必需文件；该命令仅在人工初始化或升级模型时联网。"""
    parser = argparse.ArgumentParser(description="下载本地 RAG 嵌入与精排模型")
    parser.add_argument("--model-id", default=os.getenv("LOCAL_EMBEDDING_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument(
        "--output-dir",
        default=os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "/models/bge-small-zh-v1.5"),
    )
    parser.add_argument("--revision", default=None, help="可选的模型提交版本，用于固定生产构建")
    parser.add_argument(
        "--skip-reranker", action="store_true", help="仅下载嵌入模型；知识库将使用 RRF 顺序而不精排"
    )
    arguments = parser.parse_args()

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=arguments.model_id,
        revision=arguments.revision,
        local_dir=output_dir,
        # 仅保留 safetensors 权重和 Sentence Transformers 推理所需配置，避免下载重复 .bin 权重。
        allow_patterns=["*.json", "*.txt", "*.md", "*.safetensors", "*.model", "*.py", "1_Pooling/*"],
    )
    print(f"本地嵌入模型已下载到：{output_dir}")
    if arguments.skip_reranker:
        return

    reranker_id = os.getenv("LOCAL_RERANKER_MODEL_ID", DEFAULT_RERANKER_MODEL_ID)
    reranker_dir = Path(os.getenv("LOCAL_RERANKER_MODEL_PATH", "/models/bge-reranker-base"))
    reranker_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=reranker_id,
        revision=arguments.revision,
        local_dir=reranker_dir,
        allow_patterns=["*.json", "*.txt", "*.md", "*.safetensors", "*.model", "*.py"],
    )
    print(f"本地精排模型已下载到：{reranker_dir}")


if __name__ == "__main__":
    main()
