"""按供应商创建统一的 LangChain ``BaseChatModel`` 实例。"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings


class LLMConfigurationError(ValueError):
    """模型服务配置不完整或所选适配器不可用。"""


class LLMProviderFactory:
    """将不同供应商的初始化参数收口为统一的 LangChain 聊天模型。"""

    @classmethod
    def create(cls, settings: Settings, *, temperature: float) -> BaseChatModel:
        if not settings.llm_enabled:
            raise LLMConfigurationError(
                "LLM 未启用：请配置 LLM_MODEL；除 ollama 外还需要配置 LLM_API_KEY。"
            )

        builders = {
            "openai": _create_openai_chat_model,
            "anthropic": _create_anthropic_chat_model,
            "google": _create_google_chat_model,
            "ollama": _create_ollama_chat_model,
        }
        return builders[settings.llm_provider](settings, temperature)


def _create_openai_chat_model(settings: Settings, temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    options: dict[str, object] = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "temperature": temperature,
    }
    if settings.llm_base_url:
        options["base_url"] = settings.llm_base_url
    return ChatOpenAI(**options)


def _create_anthropic_chat_model(settings: Settings, temperature: float) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as error:  # pragma: no cover - 依赖由部署环境决定
        raise LLMConfigurationError("Anthropic 适配器不可用，请安装 langchain-anthropic。") from error
    return ChatAnthropic(model=settings.llm_model, api_key=settings.llm_api_key, temperature=temperature)


def _create_google_chat_model(settings: Settings, temperature: float) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as error:  # pragma: no cover - 依赖由部署环境决定
        raise LLMConfigurationError("Google 适配器不可用，请安装 langchain-google-genai。") from error
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.llm_api_key,
        temperature=temperature,
    )


def _create_ollama_chat_model(settings: Settings, temperature: float) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as error:  # pragma: no cover - 依赖由部署环境决定
        raise LLMConfigurationError("Ollama 适配器不可用，请安装 langchain-ollama。") from error
    options: dict[str, object] = {"model": settings.llm_model, "temperature": temperature}
    if settings.llm_base_url:
        options["base_url"] = settings.llm_base_url
    return ChatOllama(**options)
