from types import SimpleNamespace

from llama_index.core.base.llms.types import TextBlock, ThinkingBlock
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
)

from app.core import ai_provider
from app.core import message_manager as _message_manager
from app.core.models import llm_manager as _llm_manager_module
from app.core.models.llm import openai as _openai_provider


def _settings(**overrides):
    base = {
        "OPENAI_USE_RESPONSES": False,
        "LLM_MODEL": "gpt-5.2",
        "QUERY_EXPANSION_MODEL": "gpt-5.4-mini",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "EMBEDDING_OUTPUT_DIMENSION": None,
        "REASONING_SUMMARY": None,
        "openai_api_key": "test-key",
        "is_openai_api_key_placeholder": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_get_llm_uses_chat_completions_when_responses_disabled(monkeypatch):
    s = _settings(OPENAI_USE_RESPONSES=False)
    monkeypatch.setattr(ai_provider, "settings", s)
    monkeypatch.setattr(_openai_provider, "settings", s)
    monkeypatch.setattr(_llm_manager_module.llm_manager, "_cache", {})
    monkeypatch.setattr(_llm_manager_module, "settings", s)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            raise AssertionError("responses class should not be used")

    monkeypatch.setattr(_openai_provider, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(_openai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm()

    assert isinstance(llm, FakeOpenAI)
    assert llm.kwargs["model"] == "gpt-5.2"
    assert llm.kwargs["api_key"] == "test-key"


def test_get_llm_uses_responses_when_enabled(monkeypatch):
    s = _settings(OPENAI_USE_RESPONSES=True)
    monkeypatch.setattr(ai_provider, "settings", s)
    monkeypatch.setattr(_openai_provider, "settings", s)
    monkeypatch.setattr(_llm_manager_module.llm_manager, "_cache", {})
    monkeypatch.setattr(_llm_manager_module, "settings", s)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            raise AssertionError("chat completions class should not be used")

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(_openai_provider, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(_openai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm()

    assert isinstance(llm, FakeOpenAIResponses)
    assert llm.kwargs["model"] == "gpt-5.2"
    assert llm.kwargs["api_key"] == "test-key"


def test_get_llm_passes_reasoning_effort_when_responses_enabled(monkeypatch):
    s = _settings(OPENAI_USE_RESPONSES=True)
    monkeypatch.setattr(ai_provider, "settings", s)
    monkeypatch.setattr(_openai_provider, "settings", s)
    monkeypatch.setattr(_llm_manager_module.llm_manager, "_cache", {})
    monkeypatch.setattr(_llm_manager_module, "settings", s)

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(_openai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm(reasoning_effort="high")

    assert isinstance(llm, FakeOpenAIResponses)
    assert llm.kwargs["reasoning_options"] == {"effort": "high"}


def test_get_llm_ignores_reasoning_effort_when_responses_disabled(monkeypatch):
    s = _settings(OPENAI_USE_RESPONSES=False)
    monkeypatch.setattr(ai_provider, "settings", s)
    monkeypatch.setattr(_openai_provider, "settings", s)
    monkeypatch.setattr(_llm_manager_module.llm_manager, "_cache", {})
    monkeypatch.setattr(_llm_manager_module, "settings", s)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(_openai_provider, "OpenAI", FakeOpenAI)

    llm = ai_provider.get_llm(reasoning_effort="high")

    assert isinstance(llm, FakeOpenAI)
    assert "reasoning_options" not in llm.kwargs


def test_normalize_reasoning_effort_defaults_for_unknown_values():
    assert ai_provider.normalize_reasoning_effort("low") == "low"
    assert ai_provider.normalize_reasoning_effort("HIGH") == "high"
    assert ai_provider.normalize_reasoning_effort("unknown") == "medium"


def test_get_embeddings_delegates_to_manager(monkeypatch):
    from unittest.mock import MagicMock
    from app.core import embedding_manager as emb_manager_mod

    monkeypatch.setattr(
        ai_provider, "settings", _settings(EMBEDDING_OUTPUT_DIMENSION=1536)
    )
    fake_instance = MagicMock()
    captured: dict = {}

    def fake_get_instance(*, model_id, api_key=None, dimensions=None):
        captured["model_id"] = model_id
        captured["api_key"] = api_key
        captured["dimensions"] = dimensions
        return fake_instance

    monkeypatch.setattr(emb_manager_mod.embedding_manager, "get_instance", fake_get_instance)

    result = ai_provider.get_embeddings()

    assert result is fake_instance
    assert captured["model_id"] == "text-embedding-3-large"
    assert captured["api_key"] is None  # manager resolves the key internally
    assert captured["dimensions"] == 1536


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
    # invoke_llm_chat delegates to llm_manager.invoke(); patch that facade.
    from app.core.models import llm_manager as llm_manager_module

    captured: dict[str, object] = {}

    def _fake_invoke(*, model_id: str, **kwargs: object) -> object:
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs
        return SimpleNamespace(message=SimpleNamespace(blocks=[TextBlock(text="ok")], content=""))

    monkeypatch.setattr(llm_manager_module.llm_manager, "invoke", _fake_invoke)

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

    assert captured["model_id"] == "gpt-5.2"
    kw = captured["kwargs"]
    assert kw["reasoning_effort"] == "high"
    assert kw["max_output_tokens"] == 256
    assert kw["prompt_cache_key"] == "cache-key"
    assert kw["prompt_cache_retention"] == "24h"
    assert kw["safety_identifier"] == "safe-id"
    assert kw["user_tag"] == "user-1"
    assert kw["timeout_seconds"] == 9.5


def test_invoke_llm_chat_ignores_responses_only_kwargs_when_disabled(monkeypatch):
    # invoke_llm_chat delegates to llm_manager.invoke(); patch that facade.
    from app.core.models import llm_manager as llm_manager_module

    captured: dict[str, object] = {}

    def _fake_invoke(*, model_id: str, **kwargs: object) -> object:
        captured["kwargs"] = kwargs
        return SimpleNamespace(message=SimpleNamespace(blocks=[], content=""))

    monkeypatch.setattr(llm_manager_module.llm_manager, "invoke", _fake_invoke)

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

    kw = captured["kwargs"]
    assert kw["max_output_tokens"] == 128
    assert kw["user_tag"] == "user-1"
    assert kw["timeout_seconds"] == 4


def test_get_llm_includes_summary_in_reasoning_options(monkeypatch):
    """reasoning_options should contain 'summary' when reasoning_summary is a valid value."""
    s = _settings(OPENAI_USE_RESPONSES=True, REASONING_SUMMARY=None)
    monkeypatch.setattr(ai_provider, "settings", s)
    monkeypatch.setattr(_openai_provider, "settings", s)
    monkeypatch.setattr(_llm_manager_module.llm_manager, "_cache", {})
    monkeypatch.setattr(_llm_manager_module, "settings", s)

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(_openai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm(reasoning_effort="medium", reasoning_summary="concise")

    assert llm.kwargs["reasoning_options"] == {"effort": "medium", "summary": "concise"}


def test_get_llm_falls_back_to_settings_summary(monkeypatch):
    """When reasoning_summary is not passed by caller, settings.REASONING_SUMMARY is used."""
    s = _settings(OPENAI_USE_RESPONSES=True, REASONING_SUMMARY="auto")
    monkeypatch.setattr(ai_provider, "settings", s)
    monkeypatch.setattr(_openai_provider, "settings", s)
    monkeypatch.setattr(_llm_manager_module.llm_manager, "_cache", {})
    monkeypatch.setattr(_llm_manager_module, "settings", s)

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(_openai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm(reasoning_effort="high")

    assert llm.kwargs["reasoning_options"] == {"effort": "high", "summary": "auto"}


def test_get_llm_omits_summary_when_none(monkeypatch):
    """No 'summary' key is added when both caller and settings provide None."""
    s = _settings(OPENAI_USE_RESPONSES=True, REASONING_SUMMARY=None)
    monkeypatch.setattr(ai_provider, "settings", s)
    monkeypatch.setattr(_openai_provider, "settings", s)
    monkeypatch.setattr(_llm_manager_module.llm_manager, "_cache", {})
    monkeypatch.setattr(_llm_manager_module, "settings", s)

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(_openai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm(reasoning_effort="low", reasoning_summary=None)

    assert llm.kwargs["reasoning_options"] == {"effort": "low"}
    assert "summary" not in llm.kwargs["reasoning_options"]


def test_get_llm_omits_summary_for_unknown_value(monkeypatch):
    """An unrecognised summary value is silently discarded."""
    s = _settings(OPENAI_USE_RESPONSES=True, REASONING_SUMMARY=None)
    monkeypatch.setattr(ai_provider, "settings", s)
    monkeypatch.setattr(_openai_provider, "settings", s)
    monkeypatch.setattr(_llm_manager_module.llm_manager, "_cache", {})
    monkeypatch.setattr(_llm_manager_module, "settings", s)

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(_openai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm(reasoning_effort="medium", reasoning_summary="verbose")

    assert "summary" not in llm.kwargs["reasoning_options"]


def test_to_chat_messages_preserves_phase_on_assistant(monkeypatch):
    """phase='commentary' on an assistant message dict is forwarded via additional_kwargs."""
    from llama_index.core.base.llms.types import MessageRole

    messages = _message_manager._to_chat_messages([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Let me think...", "phase": "commentary"},
        {"role": "assistant", "content": "Here is the answer.", "phase": "final_answer"},
    ])

    assert messages[0].role == MessageRole.USER
    assert messages[0].additional_kwargs.get("phase") is None

    assert messages[1].role == MessageRole.ASSISTANT
    assert messages[1].additional_kwargs.get("phase") == "commentary"

    assert messages[2].role == MessageRole.ASSISTANT
    assert messages[2].additional_kwargs.get("phase") == "final_answer"


def test_to_chat_messages_does_not_add_phase_to_non_assistant(monkeypatch):
    """phase is only forwarded for assistant role; ignored for user/system messages."""
    messages = _message_manager._to_chat_messages([
        {"role": "user", "content": "hi", "phase": "commentary"},
    ])

    assert not messages[0].additional_kwargs.get("phase")


def test_extract_chat_response_text_reads_raw_responses_output_text():
    response = SimpleNamespace(
        message=SimpleNamespace(blocks=[], content=""),
        raw=SimpleNamespace(output_text="Raw Responses answer."),
    )

    assert ai_provider.extract_chat_response_text(response) == "Raw Responses answer."


def test_extract_chat_response_text_reads_raw_responses_output_items():
    response = SimpleNamespace(
        message=SimpleNamespace(blocks=[], content=""),
        raw=SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(text="Object-shaped answer.")],
                )
            ]
        ),
    )

    assert (
        ai_provider.extract_chat_response_text(response) == "Object-shaped answer."
    )


def test_extract_chat_response_text_reads_dict_responses_output_items():
    response = {
        "output": [
            {
                "type": "message",
                "content": [{"text": "Dict-shaped answer."}],
            }
        ]
    }

    assert ai_provider.extract_chat_response_text(response) == "Dict-shaped answer."


# --- stream_invoke_llm_chat tests ---


def _make_stream_chunk(raw_event, delta=""):
    """Build a minimal ChatResponse-like object for streaming tests."""
    return SimpleNamespace(
        raw=raw_event,
        delta=delta,
        message=SimpleNamespace(blocks=[]),
    )


def test_stream_invoke_llm_chat_yields_reasoning_summary_deltas(monkeypatch):
    """stream_invoke_llm_chat delegates to llm_manager.stream and forwards all tuples."""
    from app.core.models import llm_manager as llm_manager_module

    def _fake_stream(*, model_id: str, **kwargs):
        yield ("Because prices", None)
        yield (None, "The answer.")

    monkeypatch.setattr(llm_manager_module.llm_manager, "stream", _fake_stream)

    results = list(
        ai_provider.stream_invoke_llm_chat(
            model="gpt-5.2",
            input_messages=[{"role": "user", "content": "question"}],
        )
    )

    reasoning_tuples = [(r, a) for r, a in results if r is not None]
    answer_tuples = [(r, a) for r, a in results if a is not None]

    assert any("Because prices" in (r or "") for r, a in reasoning_tuples)
    assert any("The answer." in (a or "") for r, a in answer_tuples)


def test_stream_invoke_llm_chat_yields_reasoning_summary_done_text(monkeypatch):
    """stream_invoke_llm_chat forwards reasoning-done tuples from llm_manager.stream."""
    from app.core.models import llm_manager as llm_manager_module

    def _fake_stream(*, model_id: str, **kwargs):
        yield ("Full summary from done event.", None)

    monkeypatch.setattr(llm_manager_module.llm_manager, "stream", _fake_stream)

    results = list(
        ai_provider.stream_invoke_llm_chat(
            model="gpt-5.2",
            input_messages=[{"role": "user", "content": "question"}],
        )
    )

    reasoning_tuples = [(r, a) for r, a in results if r is not None]
    assert reasoning_tuples == [("Full summary from done event.", None)]


def test_stream_invoke_llm_chat_handles_generic_reasoning_summary_events(monkeypatch):
    """stream_invoke_llm_chat forwards generic reasoning tuples from llm_manager.stream."""
    from app.core.models import llm_manager as llm_manager_module

    def _fake_stream(*, model_id: str, **kwargs):
        yield ("Generic summary ", None)
        yield (None, "Answer here.")

    monkeypatch.setattr(llm_manager_module.llm_manager, "stream", _fake_stream)

    results = list(
        ai_provider.stream_invoke_llm_chat(
            model="gpt-5.5",
            input_messages=[{"role": "user", "content": "question"}],
        )
    )

    reasoning_tuples = [(r, a) for r, a in results if r is not None]
    assert reasoning_tuples == [("Generic summary ", None)]


def test_stream_invoke_llm_chat_yields_raw_reasoning_text_deltas(monkeypatch):
    """stream_invoke_llm_chat forwards raw reasoning deltas from llm_manager.stream."""
    from app.core.models import llm_manager as llm_manager_module

    def _fake_stream(*, model_id: str, **kwargs):
        yield ("Raw thinking token", None)
        yield (None, "Answer here.")

    monkeypatch.setattr(llm_manager_module.llm_manager, "stream", _fake_stream)

    results = list(
        ai_provider.stream_invoke_llm_chat(
            model="gpt-5.2",
            input_messages=[{"role": "user", "content": "question"}],
        )
    )

    reasoning_tuples = [(r, a) for r, a in results if r is not None]
    assert any("Raw thinking token" in (r or "") for r, a in reasoning_tuples)


def test_stream_invoke_llm_chat_thinking_block_fallback_on_completed_event(monkeypatch):
    """stream_invoke_llm_chat forwards ThinkingBlock fallback tuples from llm_manager.stream."""
    from app.core.models import llm_manager as llm_manager_module

    def _fake_stream(*, model_id: str, **kwargs):
        yield (None, "The answer.")
        yield ("Full reasoning summary.", None)

    monkeypatch.setattr(llm_manager_module.llm_manager, "stream", _fake_stream)

    results = list(
        ai_provider.stream_invoke_llm_chat(
            model="gpt-5.2",
            input_messages=[{"role": "user", "content": "question"}],
        )
    )

    reasoning_tuples = [(r, a) for r, a in results if r is not None]
    assert len(reasoning_tuples) == 1
    assert reasoning_tuples[0][0] == "Full reasoning summary."
