import pytest

from app.config import Settings
from app.services import llm_factory


@pytest.mark.parametrize(
    ("provider", "api_key", "builder_name"),
    [
        ("openai", "test-key", "_create_openai_chat_model"),
        ("anthropic", "test-key", "_create_anthropic_chat_model"),
        ("google", "test-key", "_create_google_chat_model"),
        ("ollama", None, "_create_ollama_chat_model"),
    ],
)
def test_factory_routes_to_the_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    api_key: str | None,
    builder_name: str,
) -> None:
    expected = object()
    observed: dict[str, object] = {}

    def fake_builder(settings: Settings, temperature: float) -> object:
        observed["settings"] = settings
        observed["temperature"] = temperature
        return expected

    monkeypatch.setattr(llm_factory, builder_name, fake_builder)
    settings = Settings(llm_provider=provider, llm_model="test-model", llm_api_key=api_key)

    assert llm_factory.LLMProviderFactory.create(settings, temperature=0.2) is expected
    assert observed == {"settings": settings, "temperature": 0.2}


def test_ollama_can_be_enabled_without_an_api_key() -> None:
    assert Settings(llm_provider="ollama", llm_model="qwen3").llm_enabled is True
    assert Settings(llm_provider="openai", llm_model="gpt-test").llm_enabled is False


def test_openai_compatible_provider_preserves_current_default_behavior() -> None:
    model = llm_factory.LLMProviderFactory.create(
        Settings(llm_model="compatible-model", llm_api_key="test-key", llm_base_url="https://example.test/v1"),
        temperature=0,
    )

    assert model.__class__.__name__ == "ChatOpenAI"


def test_factory_rejects_incomplete_configuration() -> None:
    with pytest.raises(llm_factory.LLMConfigurationError, match="LLM 未启用"):
        llm_factory.LLMProviderFactory.create(Settings(), temperature=0)
