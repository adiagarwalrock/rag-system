from app.ingestion.pdf_pipeline import artifact_builders as builders


def test_run_reasoning_inference_passes_max_output_tokens(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_invoke_llm_chat(**kwargs):
        captured["max_output_tokens"] = kwargs.get("max_output_tokens")
        captured["model"] = kwargs.get("model")
        return {"id": "resp-1"}

    monkeypatch.setattr(
        builders,
        "invoke_llm_chat",
        _fake_invoke_llm_chat,
    )
    monkeypatch.setattr(
        builders,
        "extract_chat_response_text",
        lambda _response: (
            '{"key_insights":["A"],"metric_comparisons":[],"trend_statement":"T",'
            '"caveats":[],"evidence_refs":["e1"]}'
        ),
    )

    result = builders._run_reasoning_inference(
        prompt="Test prompt",
        max_output_tokens=77,
        model="gpt-5.2",
        timeout_seconds=5.0,
    )

    assert result is not None
    assert captured["max_output_tokens"] == 77
    assert captured["model"] == "gpt-5.2"
