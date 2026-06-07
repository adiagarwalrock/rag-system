from app.core.models.llm.registry import LLM_REGISTRY, LLM_REGISTRY_IDS


def test_llm_registry_contains_current_provider_models() -> None:
    assert {
        "openai/gpt-5.5",
        "openai/gpt-5.5-pro",
        "openai/gpt-5.4-mini",
        "openai/gpt-5.4-nano",
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-haiku-4-5",
        "gemini/gemini-3.1-pro-preview",
        "gemini/gemini-3.5-flash",
        "gemini/gemini-3.1-flash-lite",
    } == LLM_REGISTRY_IDS


def test_llm_registry_has_unique_ids_and_one_default() -> None:
    assert len(LLM_REGISTRY_IDS) == len(LLM_REGISTRY)
    assert [entry.id for entry in LLM_REGISTRY if entry.default] == ["openai/gpt-5.5"]
