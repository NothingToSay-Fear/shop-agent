from datetime import datetime

from pydantic import BaseModel


class KnowledgeDocumentRead(BaseModel):
    """知识库文件的列表与详情响应。"""

    id: str
    space: str
    title: str
    original_filename: str
    file_type: str
    status: str
    error_message: str | None
    chunk_count: int
    index_status: str | None = None
    index_stage: str | None = None
    processed_chunks: int = 0
    total_chunks: int = 0
    index_error_message: str | None = None
    retrieval_enabled: bool = False
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class KnowledgeDocumentContent(KnowledgeDocumentRead):
    """文件预览时额外返回已提取的纯文本。"""

    content: str


class KnowledgeDocumentRetrievalUpdate(BaseModel):
    """用户更新一份可见资料是否参与本人问答检索的请求。"""

    retrieval_enabled: bool
