from app.indexing import vector_store
from app.indexing.vector_store import VectorStoreManager


def test_configure_llama_settings_delegates_to_central_provider_init(monkeypatch):
    calls = {"count": 0}

    def _fake_initialize_ai_provider():
        calls["count"] += 1

    monkeypatch.setattr(
        vector_store,
        "initialize_ai_provider",
        _fake_initialize_ai_provider,
    )

    manager = VectorStoreManager()
    manager.configure_llama_settings()

    assert calls["count"] == 1
