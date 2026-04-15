from types import SimpleNamespace

from app.retrieval import query_expansion


def test_build_query_variants_uses_structured_output_and_enforces_dedupe_and_cap(
    monkeypatch,
):
    monkeypatch.setattr(
        query_expansion,
        "settings",
        SimpleNamespace(
            ai_api_key="test-openai-key",
            QUERY_EXPANSION_MODEL="gpt-4o-mini",
            OPENAI_USE_RESPONSES=False,
        ),
    )

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def structured_predict(self, output_cls, prompt, **prompt_args):
            self.calls.append((output_cls, prompt, prompt_args))
            return output_cls(
                rewrites=[
                    "Compare the current and older versions",
                    "Key differences between current and older versions",
                    "key differences between current and older versions",
                    "Version delta summary for current vs older",
                    "Unreached rewrite due to cap",
                ]
            )

    fake_llm = FakeLLM()
    monkeypatch.setattr(query_expansion, "get_llm", lambda **_: fake_llm)

    question = "Compare the current and older versions"
    variants = query_expansion.build_query_variants(question, max_rewrites=2)

    assert variants == [
        question,
        "Key differences between current and older versions",
        "Version delta summary for current vs older",
    ]

    assert len(fake_llm.calls) == 1
    _, _, prompt_args = fake_llm.calls[0]
    assert prompt_args["question"] == question
    assert prompt_args["max_rewrites"] == 2


def test_build_query_variants_falls_back_to_original_on_structured_error(monkeypatch):
    monkeypatch.setattr(
        query_expansion,
        "settings",
        SimpleNamespace(
            ai_api_key="test-openai-key",
            QUERY_EXPANSION_MODEL="gpt-4o-mini",
            OPENAI_USE_RESPONSES=False,
        ),
    )

    class FailingLLM:
        def structured_predict(self, output_cls, prompt, **prompt_args):
            raise RuntimeError("simulated structured prediction failure")

    monkeypatch.setattr(query_expansion, "get_llm", lambda **_: FailingLLM())

    question = "Compare versions across quarterly reports"
    assert query_expansion.build_query_variants(question) == [question]


def test_build_query_variants_skips_llm_for_non_expansion_queries(monkeypatch):
    class ShouldNotBeCalledLLM:
        def structured_predict(self, output_cls, prompt, **prompt_args):
            raise AssertionError("LLM should not be called for narrow queries")

    monkeypatch.setattr(query_expansion, "get_llm", lambda **_: ShouldNotBeCalledLLM())

    question = "What is the renewal deadline?"
    assert query_expansion.build_query_variants(question) == [question]


def test_build_query_variants_skips_llm_for_empty_question(monkeypatch):

    class ShouldNotBeCalledLLM:
        def structured_predict(self, output_cls, prompt, **prompt_args):
            raise AssertionError("LLM should not be called for empty question")

    monkeypatch.setattr(query_expansion, "get_llm", lambda **_: ShouldNotBeCalledLLM())

    question = "   "
    assert query_expansion.build_query_variants(question) == [""]


def test_build_query_variants_builds_llm_with_query_expansion_model(monkeypatch):
    monkeypatch.setattr(
        query_expansion,
        "settings",
        SimpleNamespace(
            ai_api_key="test-openai-key",
            QUERY_EXPANSION_MODEL="gpt-5.4-mini",
            OPENAI_USE_RESPONSES=True,
        ),
    )

    captured = {}

    class FakeLLM:
        def structured_predict(self, output_cls, prompt, **prompt_args):
            return output_cls(rewrites=["Compare old and new"])

    def _fake_get_llm(**kwargs):
        captured.update(kwargs)
        return FakeLLM()

    monkeypatch.setattr(query_expansion, "get_llm", _fake_get_llm)

    question = "Compare quarterly trends"
    variants = query_expansion.build_query_variants(question)

    assert variants == [question, "Compare old and new"]
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["api_key"] == "test-openai-key"
