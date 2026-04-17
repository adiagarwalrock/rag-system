from types import SimpleNamespace

from app.indexing import chat_history_store as store_module
from app.indexing.chat_history_store import ChatHistoryVectorStore


def test_search_uses_query_points_when_available(monkeypatch):
    class FakeEmbed:
        def get_text_embedding(self, text: str):
            return [0.1, 0.2, 0.3]

    class FakeClient:
        def __init__(self):
            self.called = False

        def query_points(self, **kwargs):
            self.called = True
            point = SimpleNamespace(
                score=0.88,
                payload={
                    "client_id": "client-1",
                    "session_id": "session-2",
                    "user_text": "How does renewal work?",
                    "assistant_text": "Renewals are annual.",
                    "assistant_message_id": "msg-1",
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
            )
            return SimpleNamespace(points=[point])

    fake_client = FakeClient()

    monkeypatch.setattr(
        store_module.vector_store_manager,
        "get_qdrant_client",
        lambda: fake_client,
    )

    store = ChatHistoryVectorStore()
    store._embed_model = FakeEmbed()
    store._collection_checked = True

    matches = store.search(
        client_id="client-1",
        query_text="renewal terms",
        limit=3,
        exclude_session_id="session-current",
    )

    assert fake_client.called is True
    assert len(matches) == 1
    assert matches[0].score == 0.88
    assert matches[0].session_id == "session-2"
