import typing
from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import MessageRole, TextBlock

from app.agents import agent_base as ai_provider
from app.agents.tools import provider_factory as ai_factory


def _settings(**overrides):
    base = {
        "OPENAI_USE_RESPONSES": False,
        "LLM_MODEL": "gpt-5.2",
        "QUERY_EXPANSION_MODEL": "gpt-5.4-mini",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "EMBEDDING_OUTPUT_DIMENSION": None,
        "ai_api_key": "fallback-key",
        "openai_api_key": "test-openai-key",
        "gemini_api_key": "",
        "google_api_key": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_provider_resolver_prefers_gemini_keys_over_openai_key():
    resolver = ai_factory.AIProviderFactoryResolver(
        openai_api_key="openai-key",
        gemini_api_key="gemini-key",
        google_api_key="",
    )

    provider = resolver.resolve_provider(model="gpt-5.2")

    assert provider == "gemini"


def test_provider_resolver_falls_back_to_model_when_no_keys_present():
    resolver = ai_factory.AIProviderFactoryResolver(
        openai_api_key="",
        gemini_api_key="",
        google_api_key="",
    )

    assert resolver.resolve_provider(model="gemini-2.5-flash") == "gemini"
    assert resolver.resolve_provider(model="gpt-5.2") == "openai"


def test_get_llm_uses_chat_completions_when_responses_disabled(monkeypatch):
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=False))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            raise AssertionError("responses class should not be used")

    monkeypatch.setattr(ai_factory, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(ai_factory, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm()

    assert isinstance(llm, FakeOpenAI)
    assert llm.kwargs["model"] == "gpt-5.2"
    assert llm.kwargs["api_key"] == "test-openai-key"


def test_get_llm_uses_responses_when_enabled(monkeypatch):
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=True))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            raise AssertionError("chat completions class should not be used")

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ai_factory, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(ai_factory, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm(reasoning_effort="high")

    assert isinstance(llm, FakeOpenAIResponses)
    assert llm.kwargs["model"] == "gpt-5.2"
    assert llm.kwargs["api_key"] == "test-openai-key"
    assert llm.kwargs["reasoning_options"] == {"effort": "high"}


def test_get_llm_uses_gemini_factory_when_gemini_key_present(monkeypatch):
    monkeypatch.setattr(
        ai_provider,
        "settings",
        _settings(
            LLM_MODEL="gemini-2.5-flash",
            openai_api_key="",
            gemini_api_key="gemini-key",
        ),
    )

    class FakeGoogleGenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ai_factory, "GoogleGenAI", FakeGoogleGenAI)

    llm = ai_provider.get_llm(reasoning_effort="medium")

    assert isinstance(llm, FakeGoogleGenAI)
    assert llm.kwargs["model"] == "gemini-2.5-flash"
    assert llm.kwargs["api_key"] == "gemini-key"
    generation_config = llm.kwargs["generation_config"]
    assert generation_config.thinking_config.thinking_budget == 2048


def test_normalize_reasoning_effort_defaults_for_unknown_values():
    assert ai_provider.normalize_reasoning_effort("low") == "low"
    assert ai_provider.normalize_reasoning_effort("HIGH") == "high"
    assert ai_provider.normalize_reasoning_effort("unknown") == "medium"


def test_get_embeddings_passes_optional_dimensions_for_openai(monkeypatch):
    monkeypatch.setattr(
        ai_provider,
        "settings",
        _settings(EMBEDDING_OUTPUT_DIMENSION=1536),
    )

    class FakeEmbedding:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ai_factory, "OpenAIEmbedding", FakeEmbedding)

    embedding = ai_provider.get_embeddings()

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.kwargs["model"] == "text-embedding-3-large"
    assert embedding.kwargs["api_key"] == "test-openai-key"
    assert embedding.kwargs["dimensions"] == 1536


def test_get_embeddings_passes_output_dimensionality_for_gemini(monkeypatch):
    monkeypatch.setattr(
        ai_provider,
        "settings",
        _settings(
            EMBEDDING_MODEL="gemini-embedding-2-preview",
            EMBEDDING_OUTPUT_DIMENSION=768,
            openai_api_key="",
            gemini_api_key="gemini-key",
        ),
    )

    class FakeGoogleEmbedding:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ai_factory, "GoogleGenAIEmbedding", FakeGoogleEmbedding)

    embedding = ai_provider.get_embeddings()

    assert isinstance(embedding, FakeGoogleEmbedding)
    assert embedding.kwargs["model_name"] == "gemini-embedding-2-preview"
    assert embedding.kwargs["api_key"] == "gemini-key"
    assert embedding.kwargs["embedding_config"] == {"output_dimensionality": 768}


def test_initialize_ai_provider_sets_llama_settings_and_uses_cache(monkeypatch):
    monkeypatch.setattr(ai_provider, "settings", _settings())
    fake_llama_settings = SimpleNamespace(llm=None, embed_model=None)
    monkeypatch.setattr(ai_provider, "LlamaSettings", fake_llama_settings)

    calls = {"llm": 0, "embedding": 0}

    def _fake_get_llm(**kwargs):
        calls["llm"] += 1
        return "llm-object"

    def _fake_get_embedding(**kwargs):
        calls["embedding"] += 1
        return "embed-object"

    monkeypatch.setattr(ai_provider, "get_llm", _fake_get_llm)
    monkeypatch.setattr(ai_provider, "get_embeddings", _fake_get_embedding)
    monkeypatch.setattr(ai_provider, "_CONFIGURED_SIGNATURE", None)

    ai_provider.initialize_ai_provider()
    ai_provider.initialize_ai_provider()

    assert fake_llama_settings.llm == "llm-object"
    assert fake_llama_settings.embed_model == "embed-object"
    assert calls["llm"] == 1
    assert calls["embedding"] == 1


def test_initialize_ai_provider_rejects_placeholder_key(monkeypatch):
    monkeypatch.setattr(
        ai_provider,
        "settings",
        _settings(
            openai_api_key="",
            ai_api_key="your_api_key_here",
        ),
    )
    fake_llama_settings = SimpleNamespace(llm=None, embed_model=None)
    monkeypatch.setattr(ai_provider, "LlamaSettings", fake_llama_settings)
    monkeypatch.setattr(ai_provider, "_CONFIGURED_SIGNATURE", None)

    with pytest.raises(RuntimeError, match="required"):
        ai_provider.initialize_ai_provider()


def test_invoke_llm_chat_forwards_openai_responses_runtime_kwargs(monkeypatch):
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=True))

    captured: dict[str, typing.Any] = {}

    class FakeLLM:
        def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured["chat_kwargs"] = kwargs
            return SimpleNamespace(
                message=SimpleNamespace(blocks=[TextBlock(text="ok")], content="")
            )

    def _fake_get_llm(**kwargs):
        captured["get_llm_kwargs"] = kwargs
        return FakeLLM()

    monkeypatch.setattr(ai_provider, "get_llm", _fake_get_llm)

    ai_provider.invoke_llm_chat(
        model="gpt-5.2",
        input_messages=[
            {"role": "developer", "content": "dev"},
            {"role": "user", "content": "question"},
        ],
        reasoning_effort="high",
        max_output_tokens=256,
        prompt_cache_key="cache-key",
        prompt_cache_retention="24h",
        safety_identifier="safe-id",
        user_tag="user-1",
        timeout_seconds=9.5,
    )

    assert captured["get_llm_kwargs"] == {
        "model": "gpt-5.2",
        "reasoning_effort": "high",
    }
    messages = captured["messages"]
    assert messages[0].role == MessageRole.DEVELOPER
    chat_kwargs = captured["chat_kwargs"]
    assert chat_kwargs["max_output_tokens"] == 256
    assert chat_kwargs["prompt_cache_key"] == "cache-key"
    assert chat_kwargs["prompt_cache_retention"] == "24h"
    assert chat_kwargs["safety_identifier"] == "safe-id"
    assert chat_kwargs["user"] == "user-1"
    assert chat_kwargs["timeout"] == 9.5
    assert chat_kwargs["truncation"] == "disabled"


def test_invoke_llm_chat_maps_runtime_and_roles_for_gemini(monkeypatch):
    monkeypatch.setattr(
        ai_provider,
        "settings",
        _settings(
            LLM_MODEL="gemini-2.5-flash",
            OPENAI_USE_RESPONSES=True,
            openai_api_key="",
            gemini_api_key="gemini-key",
        ),
    )

    captured: dict[str, typing.Any] = {}

    class FakeLLM:
        def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured["chat_kwargs"] = kwargs
            return SimpleNamespace(message=SimpleNamespace(blocks=[], content=""))

    monkeypatch.setattr(ai_provider, "get_llm", lambda **kwargs: FakeLLM())

    ai_provider.invoke_llm_chat(
        model="gemini-2.5-flash",
        input_messages=[
            {"role": "developer", "content": "dev"},
            {"role": "user", "content": "hello"},
        ],
        reasoning_effort="low",
        max_output_tokens=128,
        prompt_cache_key="projects/demo/locations/us/cachedContents/abc",
        prompt_cache_retention="24h",
        safety_identifier="safe-id",
        user_tag="user-1",
        timeout_seconds=4,
    )

    messages = captured["messages"]
    assert messages[0].role == MessageRole.SYSTEM
    chat_kwargs = captured["chat_kwargs"]
    assert chat_kwargs["generation_config"]["max_output_tokens"] == 128
    assert (
        chat_kwargs["generation_config"]["cached_content"]
        == "projects/demo/locations/us/cachedContents/abc"
    )
    assert "timeout" not in chat_kwargs
    assert "user" not in chat_kwargs
    assert "prompt_cache_key" not in chat_kwargs
