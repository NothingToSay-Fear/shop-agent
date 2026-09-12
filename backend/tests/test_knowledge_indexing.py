"""验证异步资料索引对前端暴露的最小任务进度契约。"""

from datetime import UTC, datetime
from types import SimpleNamespace

from app.api.knowledge import _serialize_document


def test_document_response_includes_latest_index_job_progress() -> None:
    """前端应能从同一资料响应中得到排队/处理进度与失败原因。"""
    timestamp = datetime.now(UTC)
    document = SimpleNamespace(
        id="document-1",
        space="private",
        title="618 规则",
        original_filename="618.md",
        file_type="md",
        status="processing",
        error_message=None,
        chunk_count=0,
        created_at=timestamp,
        updated_at=timestamp,
    )
    job = SimpleNamespace(
        status="running",
        stage="embedding",
        processed_chunks=32,
        total_chunks=80,
        error_message=None,
    )

    result = _serialize_document(document, job, retrieval_enabled=True)

    assert result.status == "processing"
    assert result.index_status == "running"
    assert result.index_stage == "embedding"
    assert result.processed_chunks == 32
    assert result.total_chunks == 80
    assert result.retrieval_enabled is True
