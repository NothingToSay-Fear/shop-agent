from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """集中管理运行配置，并通过环境变量决定模型设置。"""

    app_name: str = "Shop Agent API"
    database_url: str = "postgresql+asyncpg://shop_agent:change-me@db:5432/shop_agent"
    llm_provider: Literal["openai", "anthropic", "google", "ollama"] = "openai"
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    local_embedding_model_path: str | None = None
    local_embedding_model_id: str = "BAAI/bge-small-zh-v1.5"
    local_embedding_device: str = "cpu"
    local_reranker_model_path: str | None = None
    local_reranker_model_id: str = "BAAI/bge-reranker-base"
    local_reranker_device: str = "cpu"
    knowledge_reranker_min_score: float = Field(default=0.35, ge=0, le=1)
    rag_query_expansion_max_queries: int = 2
    knowledge_upload_dir: str = "/uploads"
    knowledge_max_upload_size_mb: int = 20
    knowledge_index_batch_size: int = Field(default=32, ge=1, le=256)
    knowledge_chunk_semantic_similarity_threshold: float = Field(default=0.55, ge=-1, le=1)
    knowledge_index_poll_seconds: float = Field(default=1.0, ge=0.1, le=30.0)
    knowledge_index_max_attempts: int = Field(default=3, ge=1, le=10)
    conversation_summary_poll_seconds: float = Field(default=1.0, ge=0.1, le=30.0)
    conversation_summary_max_attempts: int = Field(default=3, ge=1, le=10)
    conversation_summary_lease_seconds: int = Field(default=300, ge=30, le=3600)
    conversation_memory_token_budget: int = Field(default=12000, ge=1000, le=20000)
    conversation_memory_compact_threshold: int = Field(default=9600, ge=800, le=19000)
    conversation_memory_recent_message_limit: int = Field(default=6, ge=2, le=12)
    user_memory_focus_direction_ttl_days: int = Field(default=90, ge=7, le=365)
    user_memory_max_active_records: int = Field(default=50, ge=5, le=500)
    web_search_provider: str = "tavily"
    web_search_api_key: str | None = None
    web_search_max_results: int = 5
    web_search_timeout_seconds: float = 10.0
    auth_token_ttl_days: int = 7
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        # Docker 与本地开发可能以逗号分隔的单个字符串传入多个来源。
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def llm_enabled(self) -> bool:
        # Ollama 可在本地无密钥运行；其余已支持提供商仍必须同时具备模型名与密钥。
        return bool(self.llm_model and (self.llm_provider == "ollama" or self.llm_api_key))

    @property
    def local_embedding_enabled(self) -> bool:
        """仅在配置本地模型目录后启用本地语义向量化。"""
        return bool(self.local_embedding_model_path)

    @property
    def local_reranker_enabled(self) -> bool:
        """仅在配置本地 CrossEncoder 目录后启用知识库精排。"""
        return bool(self.local_reranker_model_path)

    @property
    def web_search_enabled(self) -> bool:
        """仅在明确选择已支持的提供方并配置密钥后发起外部网络请求。"""
        return self.web_search_provider.lower() == "tavily" and bool(self.web_search_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
