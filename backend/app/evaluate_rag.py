"""执行隔离环境中的真实 RAG 评测并生成报告。"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from app.config import get_settings
from app.evaluation.dataset import load_dataset
from app.evaluation.runner import prepare_evaluation_corpus, run_evaluation, write_baseline, write_report


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Shop Agent 真实 RAG 离线评测")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "v1",
        help="版本化评测集目录",
    )
    parser.add_argument("--prepare", action="store_true", help="写入隔离评测语料并建立索引")
    parser.add_argument("--write-baseline", action="store_true", help="以本次结果更新数据集基线文件")
    parser.add_argument("--output-dir", type=Path, default=Path("/reports"), help="报告输出目录")
    return parser.parse_args()


async def _main(arguments: argparse.Namespace) -> int:
    settings = get_settings()
    dataset = load_dataset(arguments.dataset)
    if arguments.prepare:
        allowed = "evaluation" in settings.database_url or os.getenv("EVALUATION_ALLOW_DATABASE_WRITE") == "true"
        if not allowed:
            raise RuntimeError(
                "拒绝向非评测数据库写入语料；请使用 evaluation Docker Profile，或显式设置 EVALUATION_ALLOW_DATABASE_WRITE=true"
            )
        await prepare_evaluation_corpus(dataset, settings)
    report = await run_evaluation(dataset, settings)
    json_path, markdown_path = write_report(report, arguments.output_dir)
    print(f"JSON 报告：{json_path}")
    print(f"Markdown 报告：{markdown_path}")
    if arguments.write_baseline:
        print(f"已写入基线：{write_baseline(report, dataset.root)}")
    if not report["passed"]:
        print("RAG 评测未通过质量门禁或基线回归检查。")
        return 1
    print("RAG 评测通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_parse_arguments())))
