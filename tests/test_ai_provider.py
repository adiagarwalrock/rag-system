from types import SimpleNamespace

from llama_index.core.base.llms.types import TextBlock

from app.core import ai_provider


def _settings(**overrides):
    base = {
        "OPENAI_USE_RESPONSES": False,
        "LLM_MODEL": "gpt-5.2",
        "QUERY_EXPANSION_MODEL": "gpt-5.4-mini",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "EMBEDDING_OUTPUT_DIMENSION": None,
        "ai_api_key": "test-key",
        "is_openai_api_key_placeholder": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_get_llm_uses_chat_completions_when_responses_disabled(monkeypatch):
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=False))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            raise AssertionError("responses class should not be used")

    monkeypatch.setattr(ai_provider, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(ai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm()

    assert isinstance(llm, FakeOpenAI)
    assert llm.kwargs["model"] == "gpt-5.2"
    assert llm.kwargs["api_key"] == "test-key"


def test_get_llm_uses_responses_when_enabled(monkeypatch):
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=True))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            raise AssertionError("chat completions class should not be used")

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ai_provider, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(ai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm()

    assert isinstance(llm, FakeOpenAIResponses)
    assert llm.kwargs["model"] == "gpt-5.2"
    assert llm.kwargs["api_key"] == "test-key"


def test_get_llm_passes_reasoning_effort_when_responses_enabled(monkeypatch):
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=True))

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm(reasoning_effort="high")

    assert isinstance(llm, FakeOpenAIResponses)
    assert llm.kwargs["reasoning_options"] == {"effort": "high"}


def test_get_llm_ignores_reasoning_effort_when_responses_disabled(monkeypatch):
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=False))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ai_provider, "OpenAI", FakeOpenAI)

    llm = ai_provider.get_llm(reasoning_effort="high")

    assert isinstance(llm, FakeOpenAI)
    assert "reasoning_options" not in llm.kwargs


def test_normalize_reasoning_effort_defaults_for_unknown_values():
    assert ai_provider.normalize_reasoning_effort("low") == "low"
    assert ai_provider.normalize_reasoning_effort("HIGH") == "high"
    assert ai_provider.normalize_reasoning_effort("unknown") == "medium"


def test_get_embeddings_passes_optional_dimensions(monkeypatch):
    monkeypatch.setattr(
        ai_provider, "settings", _settings(EMBEDDING_OUTPUT_DIMENSION=1536)
    )

    class FakeEmbedding:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ai_provider, "OpenAIEmbedding", FakeEmbedding)

    embedding = ai_provider.get_embeddings()

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.kwargs["model"] == "text-embedding-3-large"
    assert embedding.kwargs["api_key"] == "test-key"
    assert embedding.kwargs["dimensions"] == 1536


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


def test_invoke_llm_chat_forwards_responses_runtime_kwargs(monkeypatch):
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=True))

    captured: dict[str, object] = {}

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
    chat_kwargs = captured["chat_kwargs"]
    assert chat_kwargs["max_output_tokens"] == 256
    assert chat_kwargs["prompt_cache_key"] == "cache-key"
    assert chat_kwargs["prompt_cache_retention"] == "24h"
    assert chat_kwargs["safety_identifier"] == "safe-id"
    assert chat_kwargs["user"] == "user-1"
    assert chat_kwargs["timeout"] == 9.5
    assert chat_kwargs["truncation"] == "disabled"


def test_invoke_llm_chat_ignores_responses_only_kwargs_when_disabled(monkeypatch):
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=False))

    captured: dict[str, object] = {}

    class FakeLLM:
        def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured["chat_kwargs"] = kwargs
            return SimpleNamespace(message=SimpleNamespace(blocks=[], content=""))

    monkeypatch.setattr(ai_provider, "get_llm", lambda **kwargs: FakeLLM())

    ai_provider.invoke_llm_chat(
        model="gpt-5.2",
        input_messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="low",
        max_output_tokens=128,
        prompt_cache_key="cache-key",
        prompt_cache_retention="24h",
        safety_identifier="safe-id",
        user_tag="user-1",
        timeout_seconds=4,
    )

    chat_kwargs = captured["chat_kwargs"]
    assert chat_kwargs["max_tokens"] == 128
    assert chat_kwargs["user"] == "user-1"
    assert chat_kwargs["timeout"] == 4.0
    assert "prompt_cache_key" not in chat_kwargs
    assert "prompt_cache_retention" not in chat_kwargs
    assert "safety_identifier" not in chat_kwargs
    assert "max_output_tokens" not in chat_kwargs
    assert "truncation" not in chat_kwargs
