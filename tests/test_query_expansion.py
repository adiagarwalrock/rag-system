from types import SimpleNamespace

from app.services import query_rewriter as query_expansion


def _test_settings() -> SimpleNamespace:
    return SimpleNamespace(
        QUERY_EXPANSION_MODEL="gpt-5.4-mini",
        RESPONSE_PROMPT_CACHE_RETENTION="24h",
        RESPONSE_SAFETY_IDENTIFIER_PREFIX="rag-client",
        RESPONSE_USER_TAG="developer",
    )


def test_build_query_variants_uses_message_history_and_enforces_dedupe_and_cap(
    monkeypatch,
):
    monkeypatch.setattr(query_expansion, "settings", _test_settings())

    captured: dict = {}

    def _fake_invoke_llm_chat(**kwargs):
        captured.update(kwargs)
        return {"id": "resp-1"}

    monkeypatch.setattr(
        query_expansion,
        "invoke_llm_chat",
        _fake_invoke_llm_chat,
    )
    monkeypatch.setattr(
        query_expansion,
        "extract_chat_response_text",
        lambda _response: (
            '{"rewrites": ['
            '"Compare the current and older versions", '
            '"Key differences between current and older versions", '
            '"key differences between current and older versions", '
            '"Version delta summary for current vs older", '
            '"Unreached rewrite due to cap"'
            "]}"
        ),
    )

    payload = {
        "current_question": "Compare the current and older versions",
        "recent_turns": [
            {"role": "user", "content": "Show v1 retention policy."},
            {"role": "assistant", "content": "v1 retention is 30 days."},
            {"role": "user", "content": "Now show v2."},
            {"role": "assistant", "content": "v2 retention is 45 days."},
        ],
    }

    variants = query_expansion.build_query_variants(payload, max_rewrites=2)

    assert variants == [
        "Compare the current and older versions",
        "Key differences between current and older versions",
        "Version delta summary for current vs older",
    ]
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["reasoning_effort"] == "low"
    input_messages = captured["input_messages"]
    assert input_messages[0]["role"] == "developer"
    assert input_messages[1] == {"role": "user", "content": "Show v1 retention policy."}
    assert input_messages[2] == {
        "role": "assistant",
        "content": "v1 retention is 30 days.",
    }
    assert input_messages[-1]["role"] == "user"
    assert (
        "Latest question:\nCompare the current and older versions"
        in input_messages[-1]["content"]
    )


def test_build_query_variants_falls_back_to_original_on_llm_error(monkeypatch):
    monkeypatch.setattr(query_expansion, "settings", _test_settings())

    def _failing_invoke_llm_chat(**_kwargs):
        raise RuntimeError("simulated llm invocation failure")

    monkeypatch.setattr(
        query_expansion,
        "invoke_llm_chat",
        _failing_invoke_llm_chat,
    )

    question = "Compare versions across quarterly reports"
    assert query_expansion.build_query_variants(question) == [question]


def test_build_query_variants_skips_llm_for_non_expansion_queries(monkeypatch):
    def _should_not_be_called(**_kwargs):
        raise AssertionError("Responses API should not be called for narrow queries")

    monkeypatch.setattr(
        query_expansion,
        "invoke_llm_chat",
        _should_not_be_called,
    )

    question = "What is the renewal deadline?"
    assert query_expansion.build_query_variants(question) == [question]


def test_build_query_variants_uses_recent_history_for_vague_follow_up(monkeypatch):
    monkeypatch.setattr(query_expansion, "settings", _test_settings())

    captured: dict = {}

    def _fake_invoke_llm_chat(**kwargs):
        captured.update(kwargs)
        return {"id": "resp-2"}

    monkeypatch.setattr(
        query_expansion,
        "invoke_llm_chat",
        _fake_invoke_llm_chat,
    )
    monkeypatch.setattr(
        query_expansion,
        "extract_chat_response_text",
        lambda _response: '{"rewrites": ["Compare retention policy v2 against v1"]}',
    )

    payload = {
        "current_question": "What about that one?",
        "recent_turns": [
            {"role": "user", "content": "Compare v1 and v2 retention policy."},
            {
                "role": "assistant",
                "content": "v1 is 30 days and v2 is 45 days for enterprise accounts.",
            },
        ],
    }

    variants = query_expansion.build_query_variants(payload)

    assert variants == [
        "What about that one?",
        "Compare retention policy v2 against v1",
    ]
    assert query_expansion.should_expand_query(payload) is True
    input_messages = captured["input_messages"]
    assert input_messages[1]["content"] == "Compare v1 and v2 retention policy."
    assert input_messages[2]["content"].startswith("v1 is 30 days and v2 is 45 days")


def test_build_query_variants_skips_llm_for_empty_question(monkeypatch):
    def _should_not_be_called(**_kwargs):
        raise AssertionError("Responses API should not be called for empty question")

    monkeypatch.setattr(
        query_expansion,
        "invoke_llm_chat",
        _should_not_be_called,
    )

    question = "   "
    assert query_expansion.build_query_variants(question) == [""]


def test_build_query_variants_builds_with_query_expansion_model(monkeypatch):
    monkeypatch.setattr(
        query_expansion,
        "settings",
        SimpleNamespace(
            QUERY_EXPANSION_MODEL="gpt-4.1-mini",
            RESPONSE_PROMPT_CACHE_RETENTION="24h",
            RESPONSE_SAFETY_IDENTIFIER_PREFIX="rag-client",
            RESPONSE_USER_TAG="developer",
        ),
    )

    captured: dict = {}

    def _fake_invoke_llm_chat(**kwargs):
        captured.update(kwargs)
        return {"id": "resp-3"}

    monkeypatch.setattr(
        query_expansion,
        "invoke_llm_chat",
        _fake_invoke_llm_chat,
    )
    monkeypatch.setattr(
        query_expansion,
        "extract_chat_response_text",
        lambda _response: '{"rewrites": ["Compare old and new"]}',
    )

    question = "Compare quarterly trends"
    variants = query_expansion.build_query_variants(question)

    assert variants == [question, "Compare old and new"]
    assert captured["model"] == "gpt-4.1-mini"


def test_build_query_variants_limits_history_to_latest_three_pairs(monkeypatch):
    monkeypatch.setattr(query_expansion, "settings", _test_settings())

    captured: dict = {}

    def _fake_invoke_llm_chat(**kwargs):
        captured.update(kwargs)
        return {"id": "resp-4"}

    monkeypatch.setattr(
        query_expansion,
        "invoke_llm_chat",
        _fake_invoke_llm_chat,
    )
    monkeypatch.setattr(
        query_expansion,
        "extract_chat_response_text",
        lambda _response: '{"rewrites": ["Compare current release vs previous release"]}',
    )

    turns = [
        {"role": "user", "content": "turn 1 user"},
        {"role": "assistant", "content": "turn 1 assistant"},
        {"role": "user", "content": "turn 2 user"},
        {"role": "assistant", "content": "turn 2 assistant"},
        {"role": "user", "content": "turn 3 user"},
        {"role": "assistant", "content": "turn 3 assistant"},
        {"role": "user", "content": "turn 4 user"},
        {"role": "assistant", "content": "turn 4 assistant"},
    ]
    payload = {
        "current_question": "What about that?",
        "recent_turns": turns,
    }

    variants = query_expansion.build_query_variants(payload)

    assert variants == [
        "What about that?",
        "Compare current release vs previous release",
    ]
    input_messages = captured["input_messages"]
    history_messages = input_messages[1:-1]
    assert len(history_messages) == 6
    assert history_messages[0]["content"] == "turn 2 user"
    assert history_messages[1]["content"] == "turn 2 assistant"
    assert history_messages[-1]["content"] == "turn 4 assistant"
