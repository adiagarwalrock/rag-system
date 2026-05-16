from types import SimpleNamespace

from app.ingestion.pdf_pipeline.artifact import core as builders


def test_run_reasoning_inference_disables_output_cap_for_responses(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_invoke_llm_chat(**kwargs):
        captured["max_output_tokens"] = kwargs.get("max_output_tokens")
        captured["model"] = kwargs.get("model")
        captured["timeout_seconds"] = kwargs.get("timeout_seconds")
        captured["structured_output_cls"] = kwargs.get("structured_output_cls")
        return SimpleNamespace(
            raw=builders.ReasoningStructuredResponse(
                key_insights=["A"],
                trend_statement="T",
                evidence_refs=["e1"],
            )
        )

    monkeypatch.setattr(
        builders,
        "invoke_llm_chat",
        _fake_invoke_llm_chat,
    )

    monkeypatch.setattr(builders.settings, "OPENAI_USE_RESPONSES", True)

    result = builders._run_reasoning_inference(
        prompt="Test prompt",
        max_output_tokens=77,
        model="gpt-5.2",
        timeout_seconds=5.0,
    )

    assert result is not None
    assert captured["max_output_tokens"] is None
    assert captured["model"] == "gpt-5.2"
    assert captured["timeout_seconds"] == 5.0
    assert captured["structured_output_cls"] is builders.ReasoningStructuredResponse


def test_run_reasoning_inference_retries_with_larger_budget_on_incomplete_structured_output(
    monkeypatch,
):
    token_calls: list[int | None] = []
    attempts = {"count": 0}

    def _fake_invoke_llm_chat(**kwargs):
        token_calls.append(kwargs.get("max_output_tokens"))
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ValueError("Invalid JSON: EOF while parsing a string")
        return SimpleNamespace(
            raw=builders.ReasoningStructuredResponse(
                key_insights=["A"],
                trend_statement="T",
                evidence_refs=["e1"],
            )
        )

    monkeypatch.setattr(builders, "invoke_llm_chat", _fake_invoke_llm_chat)

    monkeypatch.setattr(builders.settings, "OPENAI_USE_RESPONSES", False)

    result = builders._run_reasoning_inference(
        prompt="Test prompt",
        max_output_tokens=700,
        model="gpt-5.2",
        timeout_seconds=5.0,
    )

    assert result is not None
    assert token_calls == [700, builders._reasoning_retry_output_tokens(700)]
