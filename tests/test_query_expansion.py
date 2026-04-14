from types import SimpleNamespace

from app.retrieval import query_expansion


def test_build_query_variants_uses_structured_output_and_enforces_dedupe_and_cap(
    monkeypatch,
):
    monkeypatch.setattr(
        query_expansion,
        "settings",
        SimpleNamespace(
            google_api_key="test-google-key",
            QUERY_EXPANSION_MODEL="gemini-3-flash-preview",
        ),
    )

    class FakeGoogleGenAI:
        instances = []

        def __init__(self, model: str, api_key: str):
            self.model = model
            self.api_key = api_key
            self.calls = []
            FakeGoogleGenAI.instances.append(self)

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

    monkeypatch.setattr(query_expansion, "GoogleGenAI", FakeGoogleGenAI)

    question = "Compare the current and older versions"
    variants = query_expansion.build_query_variants(question, max_rewrites=2)

    assert variants == [
        question,
        "Key differences between current and older versions",
        "Version delta summary for current vs older",
    ]

    assert len(FakeGoogleGenAI.instances) == 1
    instance = FakeGoogleGenAI.instances[0]
    assert instance.model == "gemini-3-flash-preview"
    assert instance.api_key == "test-google-key"
    assert len(instance.calls) == 1
    _, _, prompt_args = instance.calls[0]
    assert prompt_args["question"] == question
    assert prompt_args["max_rewrites"] == 2


def test_build_query_variants_falls_back_to_original_on_structured_error(monkeypatch):
    monkeypatch.setattr(
        query_expansion,
        "settings",
        SimpleNamespace(
            google_api_key="test-google-key",
            QUERY_EXPANSION_MODEL="gemini-3-flash-preview",
        ),
    )

    class FailingGoogleGenAI:
        def __init__(self, model: str, api_key: str):
            self.model = model
            self.api_key = api_key

        def structured_predict(self, output_cls, prompt, **prompt_args):
            raise RuntimeError("simulated structured prediction failure")

    monkeypatch.setattr(query_expansion, "GoogleGenAI", FailingGoogleGenAI)

    question = "Compare versions across quarterly reports"
    assert query_expansion.build_query_variants(question) == [question]


def test_build_query_variants_skips_llm_for_non_expansion_queries(monkeypatch):
    class ShouldNotBeCalledGoogleGenAI:
        def __init__(self, model: str, api_key: str):
            raise AssertionError("GoogleGenAI should not be called for narrow queries")

    monkeypatch.setattr(query_expansion, "GoogleGenAI", ShouldNotBeCalledGoogleGenAI)

    question = "What is the renewal deadline?"
    assert query_expansion.build_query_variants(question) == [question]


def test_build_query_variants_skips_llm_for_empty_question(monkeypatch):

    class ShouldNotBeCalledGoogleGenAI:
        def __init__(self, model: str, api_key: str):
            raise AssertionError("GoogleGenAI should not be called for empty question")

    monkeypatch.setattr(query_expansion, "GoogleGenAI", ShouldNotBeCalledGoogleGenAI)

    question = "   "
    assert query_expansion.build_query_variants(question) == [""]
