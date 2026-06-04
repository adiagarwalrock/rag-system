from types import SimpleNamespace

from llama_index.core.base.llms.types import TextBlock, ThinkingBlock
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
)

from app.core import ai_provider


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
        "reasoning_summary": None,
        "timeout_seconds": 9.5,
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


def test_get_llm_includes_summary_in_reasoning_options(monkeypatch):
    """reasoning_options should contain 'summary' when reasoning_summary is a valid value."""
    monkeypatch.setattr(
        ai_provider,
        "settings",
        _settings(OPENAI_USE_RESPONSES=True, REASONING_SUMMARY=None),
    )

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm(reasoning_effort="medium", reasoning_summary="concise")

    assert llm.kwargs["reasoning_options"] == {"effort": "medium", "summary": "concise"}


def test_get_llm_falls_back_to_settings_summary(monkeypatch):
    """When reasoning_summary is not passed by caller, settings.REASONING_SUMMARY is used."""
    monkeypatch.setattr(
        ai_provider,
        "settings",
        _settings(OPENAI_USE_RESPONSES=True, REASONING_SUMMARY="auto"),
    )

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm(reasoning_effort="high")

    assert llm.kwargs["reasoning_options"] == {"effort": "high", "summary": "auto"}


def test_get_llm_omits_summary_when_none(monkeypatch):
    """No 'summary' key is added when both caller and settings provide None."""
    monkeypatch.setattr(
        ai_provider,
        "settings",
        _settings(OPENAI_USE_RESPONSES=True, REASONING_SUMMARY=None),
    )

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm(reasoning_effort="low", reasoning_summary=None)

    assert llm.kwargs["reasoning_options"] == {"effort": "low"}
    assert "summary" not in llm.kwargs["reasoning_options"]


def test_get_llm_omits_summary_for_unknown_value(monkeypatch):
    """An unrecognised summary value is silently discarded."""
    monkeypatch.setattr(
        ai_provider,
        "settings",
        _settings(OPENAI_USE_RESPONSES=True, REASONING_SUMMARY=None),
    )

    class FakeOpenAIResponses:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ai_provider, "OpenAIResponses", FakeOpenAIResponses)

    llm = ai_provider.get_llm(reasoning_effort="medium", reasoning_summary="verbose")

    assert "summary" not in llm.kwargs["reasoning_options"]


def test_to_chat_messages_preserves_phase_on_assistant(monkeypatch):
    """phase='commentary' on an assistant message dict is forwarded via additional_kwargs."""
    from llama_index.core.base.llms.types import MessageRole

    messages = ai_provider._to_chat_messages([
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
    messages = ai_provider._to_chat_messages([
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
    """ResponseReasoningSummaryTextDeltaEvent deltas are emitted as reasoning tuples."""
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=True))

    summary_event = ResponseReasoningSummaryTextDeltaEvent(
        delta="Because prices",
        item_id="item-1",
        output_index=0,
        sequence_number=1,
        summary_index=0,
        type="response.reasoning_summary_text.delta",
    )
    text_chunk = _make_stream_chunk(raw_event=SimpleNamespace(), delta="The answer.")

    class FakeLLM:
        def stream_chat(self, messages, **kwargs):
            yield _make_stream_chunk(raw_event=summary_event)
            yield text_chunk

    monkeypatch.setattr(ai_provider, "get_llm", lambda **kw: FakeLLM())
    monkeypatch.setattr(ai_provider, "OpenAIResponses", FakeLLM)

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
    """ResponseReasoningSummaryTextDoneEvent text is used when no deltas arrived."""
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=True))

    done_event = ResponseReasoningSummaryTextDoneEvent(
        item_id="item-1",
        output_index=0,
        sequence_number=1,
        summary_index=0,
        text="Full summary from done event.",
        type="response.reasoning_summary_text.done",
    )

    class FakeLLM:
        def stream_chat(self, messages, **kwargs):
            yield _make_stream_chunk(raw_event=done_event)
            yield _make_stream_chunk(raw_event=SimpleNamespace(), delta="Answer here.")

    monkeypatch.setattr(ai_provider, "get_llm", lambda **kw: FakeLLM())
    monkeypatch.setattr(ai_provider, "OpenAIResponses", FakeLLM)

    results = list(
        ai_provider.stream_invoke_llm_chat(
            model="gpt-5.2",
            input_messages=[{"role": "user", "content": "question"}],
        )
    )

    reasoning_tuples = [(r, a) for r, a in results if r is not None]
    assert reasoning_tuples == [("Full summary from done event.", None)]


def test_stream_invoke_llm_chat_handles_generic_reasoning_summary_events(monkeypatch):
    """Generic raw event objects are accepted for gpt-5.x/LlamaIndex stream shapes."""
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=True))

    delta_event = SimpleNamespace(
        type="response.reasoning_summary_text.delta",
        delta="Generic summary ",
    )
    done_event = SimpleNamespace(
        type="response.reasoning_summary_text.done",
        text="Generic summary from done.",
    )

    class FakeLLM:
        def stream_chat(self, messages, **kwargs):
            yield _make_stream_chunk(raw_event=delta_event)
            yield _make_stream_chunk(raw_event=done_event)
            yield _make_stream_chunk(raw_event=SimpleNamespace(), delta="Answer here.")

    monkeypatch.setattr(ai_provider, "get_llm", lambda **kw: FakeLLM())
    monkeypatch.setattr(ai_provider, "OpenAIResponses", FakeLLM)

    results = list(
        ai_provider.stream_invoke_llm_chat(
            model="gpt-5.5",
            input_messages=[{"role": "user", "content": "question"}],
        )
    )

    reasoning_tuples = [(r, a) for r, a in results if r is not None]
    assert reasoning_tuples == [("Generic summary ", None)]


def test_stream_invoke_llm_chat_yields_raw_reasoning_text_deltas(monkeypatch):
    """ResponseReasoningTextDeltaEvent (raw CoT) is also emitted as reasoning tuples."""
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=True))

    raw_event = ResponseReasoningTextDeltaEvent(
        delta="Raw thinking token",
        item_id="item-1",
        output_index=0,
        sequence_number=1,
        content_index=0,
        type="response.reasoning_text.delta",
    )

    class FakeLLM:
        def stream_chat(self, messages, **kwargs):
            yield _make_stream_chunk(raw_event=raw_event)
            yield _make_stream_chunk(raw_event=SimpleNamespace(), delta="Answer here.")

    monkeypatch.setattr(ai_provider, "get_llm", lambda **kw: FakeLLM())
    monkeypatch.setattr(ai_provider, "OpenAIResponses", FakeLLM)

    results = list(
        ai_provider.stream_invoke_llm_chat(
            model="gpt-5.2",
            input_messages=[{"role": "user", "content": "question"}],
        )
    )

    reasoning_tuples = [(r, a) for r, a in results if r is not None]
    assert any("Raw thinking token" in (r or "") for r, a in reasoning_tuples)


def test_stream_invoke_llm_chat_thinking_block_fallback_on_completed_event(monkeypatch):
    """When no reasoning deltas arrive, ThinkingBlock from ResponseCompletedEvent is used."""
    monkeypatch.setattr(ai_provider, "settings", _settings(OPENAI_USE_RESPONSES=True))

    # Simulate a ResponseCompletedEvent with isinstance check via fake type
    class FakeCompletedEvent:
        pass

    # Patch ResponseCompletedEvent inside ai_provider to our fake class
    monkeypatch.setattr(ai_provider, "ResponseCompletedEvent", FakeCompletedEvent)

    completed_chunk = SimpleNamespace(
        raw=FakeCompletedEvent(),
        delta="",
        message=SimpleNamespace(
            blocks=[ThinkingBlock(content="Full reasoning summary.")]
        ),
    )
    answer_chunk = _make_stream_chunk(raw_event=SimpleNamespace(), delta="The answer.")

    class FakeLLM:
        def stream_chat(self, messages, **kwargs):
            yield answer_chunk
            yield completed_chunk

    monkeypatch.setattr(ai_provider, "get_llm", lambda **kw: FakeLLM())
    monkeypatch.setattr(ai_provider, "OpenAIResponses", FakeLLM)

    results = list(
        ai_provider.stream_invoke_llm_chat(
            model="gpt-5.2",
            input_messages=[{"role": "user", "content": "question"}],
        )
    )

    reasoning_tuples = [(r, a) for r, a in results if r is not None]
    assert len(reasoning_tuples) == 1
    assert reasoning_tuples[0][0] == "Full reasoning summary."
