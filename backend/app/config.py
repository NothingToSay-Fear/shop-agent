from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """集中管理运行配置，并通过环境变量决定模型设置。"""

    app_name: str = "Shop Agent API"
    database_url: str = "postgresql+asyncpg://shop_agent:change-me@db:5432/shop_agent"
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    local_embedding_model_path: str | None = None
    local_embedding_model_id: str = "BAAI/bge-small-zh-v1.5"
    local_embedding_device: str = "cpu"
    knowledge_upload_dir: str = "/uploads"
    knowledge_max_upload_size_mb: int = 20
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
        # 仅有模型名称不足以调用服务；缺少密钥时避免意外的外部请求。
        return bool(self.llm_api_key and self.llm_model)

    @property
    def local_embedding_enabled(self) -> bool:
        """仅在配置本地模型目录后启用本地语义向量化。"""
        return bool(self.local_embedding_model_path)

    @property
    def web_search_enabled(self) -> bool:
        """仅在明确选择已支持的提供方并配置密钥后发起外部网络请求。"""
        return self.web_search_provider.lower() == "tavily" and bool(self.web_search_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
