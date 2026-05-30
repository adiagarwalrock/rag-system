import pytest

from app.core.config import settings
from app.db.models import ConflictLog, QueryLog, RetrievalLog, VectorNodeRegistry
from app.services import query_service

VECTOR_COLLECTION = settings.VECTOR_COLLECTION


def test_execute_query_persists_query_retrieval_and_conflict_logs(
    db_session, seeded_entities, monkeypatch
):
    monkeypatch.setattr(settings, "ENABLE_AGENTIC_RAG", False)
    client = seeded_entities["client"]
    document = seeded_entities["document"]

    registry_row = VectorNodeRegistry(
        id="registry-1",
        document_id=document.id,
        client_id=client.id,
        vector_collection=VECTOR_COLLECTION,
        vector_node_id="node-1",
        embedding_model="test-model",
        is_active=True,
    )
    db_session.add(registry_row)
    db_session.commit()

    class FakeRetriever:
        def __init__(
            self,
            client_id: str,
            reasoning_effort: str = "medium",
            reasoning_summary: str | None = None,
            conversation_context: dict | None = None,
            reasoning_callback=None,
            answer_callback=None,
        ):
            self.client_id = client_id
            self.reasoning_effort = reasoning_effort
            self.reasoning_summary = reasoning_summary
            self.conversation_context = conversation_context

        def query(self, question: str, status_callback=None) -> dict:
            assert self.client_id == client.id
            assert self.reasoning_effort == "medium"
            assert question == "What changed in v2?"
            return {
                "answer": "Policy v2 changes renewal terms.",
                "citations": [
                    {"vector_node_id": "node-1", "score": 0.91},
                    {
                        "vector_node_id": "node-2",
                        "document_id": "explicit-doc-id",
                        "score": 0.75,
                    },
                ],
                "conflicts": [
                    {
                        "conflict_type": "numeric_conflict",
                        "summary": "Two sources disagree on retention period.",
                    }
                ],
            }

    monkeypatch.setattr(query_service, "VecteraRetriever", FakeRetriever)

    result = query_service.execute_query(
        question="What changed in v2?",
        client_id=client.id,
        db=db_session,
    )

    assert result["query_id"]
    assert result["latency_ms"] >= 0
    assert result["answer"] == "Policy v2 changes renewal terms."

    query_log = (
        db_session.query(QueryLog).filter(QueryLog.id == result["query_id"]).one()
    )
    assert query_log.status == "completed"

    retrieval_logs = (
        db_session.query(RetrievalLog)
        .filter(RetrievalLog.query_log_id == result["query_id"])
        .order_by(RetrievalLog.rank)
        .all()
    )
    assert len(retrieval_logs) == 2
    assert retrieval_logs[0].rank == 1
    assert retrieval_logs[0].document_id == document.id
    assert retrieval_logs[1].rank == 2
    assert retrieval_logs[1].document_id == "explicit-doc-id"

    conflict_logs = (
        db_session.query(ConflictLog)
        .filter(ConflictLog.query_log_id == result["query_id"])
        .all()
    )
    assert len(conflict_logs) == 1
    assert conflict_logs[0].conflict_type == "numeric_conflict"


def test_execute_query_marks_query_log_failed_on_retrieval_error(
    db_session, seeded_entities, monkeypatch
):
    monkeypatch.setattr(settings, "ENABLE_AGENTIC_RAG", False)
    client = seeded_entities["client"]

    class FailingRetriever:
        def __init__(
            self,
            client_id: str,
            reasoning_effort: str = "medium",
            reasoning_summary: str | None = None,
            conversation_context: dict | None = None,
            reasoning_callback=None,
            answer_callback=None,
        ):
            self.client_id = client_id
            self.reasoning_effort = reasoning_effort
            self.reasoning_summary = reasoning_summary
            self.conversation_context = conversation_context

        def query(self, question: str, status_callback=None) -> dict:
            raise RuntimeError("simulated retrieval failure")

    monkeypatch.setattr(query_service, "VecteraRetriever", FailingRetriever)

    with pytest.raises(RuntimeError, match="simulated retrieval failure"):
        query_service.execute_query(
            question="Why did this fail?",
            client_id=client.id,
            db=db_session,
        )

    query_logs = db_session.query(QueryLog).all()
    assert len(query_logs) == 1
    assert query_logs[0].status == "failed"
    assert "simulated retrieval failure" in query_logs[0].answer
