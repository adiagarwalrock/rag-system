from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.status_service import RuntimeStatusService


class FakeSettings:
    def __init__(self, **overrides):
        values = {
            "ENABLE_EXTERNAL_PARSER": True,
            "REDUCTO_API_KEY": None,
            "LLAMAPARSE_API_KEY": None,
            "QDRANT_URL": "http://qdrant:6333",
            "COLLECTION_NAME": "docs",
            "CHAT_HISTORY_COLLECTION_NAME": "chat",
            "SNOWFLAKE_ACCOUNT": None,
            "SNOWFLAKE_USER": None,
            "SNOWFLAKE_DATABASE": None,
            "SNOWFLAKE_SCHEMA": None,
            "SNOWFLAKE_WAREHOUSE": None,
            "SNOWFLAKE_ROLE": None,
            "LLM_MODEL": "gpt-test",
            "EMBEDDING_MODEL": "text-embedding-test",
            "EMBEDDING_OUTPUT_DIMENSION": None,
            "VECTOR_DIMENSIONS": 1536,
            "OPENAI_USE_RESPONSES": True,
            "ai_api_key": "test-key",
            "is_openai_api_key_placeholder": False,
        }
        values.update(overrides)
        self.__dict__.update(values)

    @property
    def effective_vector_dimensions(self):
        return self.EMBEDDING_OUTPUT_DIMENSION or self.VECTOR_DIMENSIONS


class FakeEngine:
    def __init__(self, dialect_name="sqlite", database="rag_local.db"):
        self.dialect = SimpleNamespace(name=dialect_name)
        self.url = SimpleNamespace(database=database)


class FakeSession:
    def __init__(self, error: Exception | None = None):
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, _statement):
        if self.error:
            raise self.error
        return 1


def _service(**kwargs):
    defaults = {
        "settings_obj": FakeSettings(),
        "engine_obj": FakeEngine(),
        "session_factory": lambda: FakeSession(),
        "qdrant_client_factory": lambda: FakeQdrantClient(),
        "openai_client_factory": lambda **_kwargs: FakeOpenAIClient(),
        "ai_initializer": lambda: None,
    }
    defaults.update(kwargs)
    return RuntimeStatusService(**defaults)


class FakeQdrantClient:
    def __init__(self, collections=("docs", "chat", "scratch"), fail=False):
        self.collections = collections
        self.fail = fail

    def get_collections(self):
        if self.fail:
            raise RuntimeError("qdrant unavailable")
        return SimpleNamespace(
            collections=[
                SimpleNamespace(name=collection_name)
                for collection_name in self.collections
            ]
        )

    def count(self, collection_name: str, exact: bool = False):
        assert exact is False
        return SimpleNamespace(count=len(collection_name))


class FakeModels:
    def __init__(self, model_ids=None, error: Exception | None = None):
        self.model_ids = model_ids or ["gpt-test", "text-embedding-test"]
        self.error = error

    def list(self):
        if self.error:
            raise self.error
        return SimpleNamespace(
            data=[SimpleNamespace(id=model_id) for model_id in self.model_ids]
        )


class FakeOpenAIClient:
    def __init__(self, model_ids=None, error: Exception | None = None):
        self.models = FakeModels(model_ids=model_ids, error=error)


@pytest.mark.parametrize(
    ("external_enabled", "reducto_key", "llamaparse_key", "enabled"),
    [
        (False, "red-key", "llama-key", {"Reducto": False, "LlamaParse": False}),
        (True, "red-key", None, {"Reducto": True, "LlamaParse": False}),
        (True, None, "llama-key", {"Reducto": False, "LlamaParse": True}),
        (True, "red-key", "llama-key", {"Reducto": True, "LlamaParse": True}),
    ],
)
def test_parser_status_combinations(
    external_enabled,
    reducto_key,
    llamaparse_key,
    enabled,
):
    settings = FakeSettings(
        ENABLE_EXTERNAL_PARSER=external_enabled,
        REDUCTO_API_KEY=reducto_key,
        LLAMAPARSE_API_KEY=llamaparse_key,
    )

    status = _service(settings_obj=settings).get_status()["parsers"]
    rows = {row["name"]: row for row in status["parsers"]}

    assert rows["External parser"]["enabled"] is external_enabled
    assert rows["Reducto"]["enabled"] is enabled["Reducto"]
    assert rows["LlamaParse"]["enabled"] is enabled["LlamaParse"]


def test_qdrant_status_lists_collections_with_counts_and_roles():
    status = _service().get_status()["qdrant"]

    assert status["status"] == "ok"
    assert status["url"] == "http://qdrant:6333"
    assert status["collections"] == [
        {"name": "docs", "point_count": 4, "role": "documents"},
        {"name": "chat", "point_count": 4, "role": "chat_history"},
        {"name": "scratch", "point_count": 7, "role": "other"},
    ]


def test_qdrant_status_returns_error_without_raising():
    status = _service(
        qdrant_client_factory=lambda: FakeQdrantClient(fail=True)
    ).get_status()["qdrant"]

    assert status["status"] == "error"
    assert "qdrant unavailable" in status["message"]


def test_database_status_reports_local_sqlite_success():
    status = _service().get_status()["database"]

    assert status["status"] == "ok"
    assert status["mode"] == "local_sqlite"
    assert status["dialect"] == "sqlite"
    assert status["target"] == {"path": "rag_local.db"}


def test_database_status_reports_snowflake_and_failure():
    settings = FakeSettings(
        SNOWFLAKE_ACCOUNT="acct",
        SNOWFLAKE_USER="user",
        SNOWFLAKE_DATABASE="db",
        SNOWFLAKE_SCHEMA="schema",
        SNOWFLAKE_WAREHOUSE="wh",
        SNOWFLAKE_ROLE="role",
    )
    status = _service(
        settings_obj=settings,
        engine_obj=FakeEngine(dialect_name="snowflake"),
        session_factory=lambda: FakeSession(RuntimeError("db down")),
    ).get_status()["database"]

    assert status["status"] == "error"
    assert status["mode"] == "snowflake"
    assert status["dialect"] == "snowflake"
    assert status["target"]["account"] == "acct"
    assert "db down" in status["message"]


def test_ai_status_skips_live_checks_when_key_is_missing_or_placeholder():
    settings = FakeSettings(ai_api_key="", is_openai_api_key_placeholder=True)

    status = _service(settings_obj=settings).get_status()["ai"]

    assert status["status"] == "error"
    assert status["key_valid"] is False
    assert status["initialization"]["status"] == "skipped"
    assert status["models_api"]["status"] == "skipped"


def test_ai_status_reports_models_api_success_and_model_availability():
    status = _service().get_status()["ai"]

    assert status["status"] == "ok"
    assert status["initialization"]["status"] == "ok"
    assert status["models_api"]["status"] == "ok"
    assert status["models_api"]["model_count"] == 2
    assert status["models_api"]["llm_model_available"] is True
    assert status["models_api"]["embedding_model_available"] is True


def test_ai_status_reports_initialization_failure():
    status = _service(
        ai_initializer=lambda: (_ for _ in ()).throw(RuntimeError("init failed"))
    ).get_status()["ai"]

    assert status["status"] == "error"
    assert status["initialization"]["status"] == "error"
    assert "init failed" in status["initialization"]["message"]


def test_ai_status_reports_models_api_failure():
    status = _service(
        openai_client_factory=lambda **_kwargs: FakeOpenAIClient(
            error=TimeoutError("models timeout")
        )
    ).get_status()["ai"]

    assert status["status"] == "error"
    assert status["models_api"]["status"] == "error"
    assert "models timeout" in status["models_api"]["message"]


def test_ai_status_reports_missing_configured_model_ids():
    status = _service(
        openai_client_factory=lambda **_kwargs: FakeOpenAIClient(
            model_ids=["other-model"]
        )
    ).get_status()["ai"]

    assert status["status"] == "ok"
    assert status["models_api"]["llm_model_available"] is False
    assert status["models_api"]["embedding_model_available"] is False
