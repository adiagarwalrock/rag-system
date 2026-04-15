import pytest

from app.db.models import (
    Client,
    ConflictLog,
    Document,
    DocumentVersion,
    IngestionJob,
    QueryLog,
    RetrievalLog,
    UserClientAccess,
    VectorNodeRegistry,
)
from app.services import client_service


def test_delete_client_cascades_documents_vectors_and_related_rows(
    db_session, seeded_entities, monkeypatch
):
    user = seeded_entities["user"]
    target_client = seeded_entities["client"]
    first_doc = seeded_entities["document"]

    second_doc = Document(
        id="doc-2",
        client_id=target_client.id,
        name="policy_v2.pdf",
        file_type=".pdf",
        storage_path="/tmp/policy_v2.pdf",
        status="completed",
    )
    untouched_client = Client(
        id="client-keep",
        name="Keep Co",
        description="Must remain",
        created_by=user.id,
        is_active=True,
    )
    untouched_doc = Document(
        id="doc-keep",
        client_id=untouched_client.id,
        name="retain.pdf",
        file_type=".pdf",
        storage_path="/tmp/retain.pdf",
        status="completed",
    )
    db_session.add_all([second_doc, untouched_client, untouched_doc])

    db_session.add_all(
        [
            DocumentVersion(id="ver-1", document_id=first_doc.id),
            DocumentVersion(id="ver-2", document_id=second_doc.id),
            IngestionJob(
                id="job-1",
                client_id=target_client.id,
                document_id=first_doc.id,
                status="completed",
            ),
            IngestionJob(
                id="job-2",
                client_id=target_client.id,
                document_id=second_doc.id,
                status="completed",
            ),
            IngestionJob(
                id="job-keep",
                client_id=untouched_client.id,
                document_id=untouched_doc.id,
                status="completed",
            ),
            VectorNodeRegistry(
                id="reg-1",
                document_id=first_doc.id,
                client_id=target_client.id,
                vector_collection="test-collection",
                vector_node_id="node-1",
            ),
            VectorNodeRegistry(
                id="reg-2",
                document_id=second_doc.id,
                client_id=target_client.id,
                vector_collection="test-collection",
                vector_node_id="node-2",
            ),
            VectorNodeRegistry(
                id="reg-keep",
                document_id=untouched_doc.id,
                client_id=untouched_client.id,
                vector_collection="test-collection",
                vector_node_id="node-keep",
            ),
            QueryLog(
                id="query-target",
                user_id=user.id,
                client_id=target_client.id,
                question="Target question",
                status="completed",
            ),
            RetrievalLog(
                id="retrieval-target",
                query_log_id="query-target",
                vector_node_id="node-1",
                document_id=first_doc.id,
                rank=1,
            ),
            ConflictLog(
                id="conflict-target",
                query_log_id="query-target",
                conflict_type="numeric_conflict",
            ),
            QueryLog(
                id="query-keep",
                user_id=user.id,
                client_id=untouched_client.id,
                question="Keep question",
                status="completed",
            ),
            RetrievalLog(
                id="retrieval-keep",
                query_log_id="query-keep",
                vector_node_id="node-keep",
                document_id=untouched_doc.id,
                rank=1,
            ),
            ConflictLog(
                id="conflict-keep",
                query_log_id="query-keep",
                conflict_type="policy_conflict",
            ),
            UserClientAccess(
                user_id=user.id,
                client_id=target_client.id,
                granted_by=user.id,
            ),
            UserClientAccess(
                user_id=user.id,
                client_id=untouched_client.id,
                granted_by=user.id,
            ),
        ]
    )
    db_session.commit()

    deleted_docs: list[tuple[str, bool]] = []

    def _fake_delete_document(document_id: str, db, hard: bool = False):
        deleted_docs.append((document_id, hard))
        db.query(VectorNodeRegistry).filter(
            VectorNodeRegistry.document_id == document_id
        ).delete()
        db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id
        ).delete()
        db.query(IngestionJob).filter(IngestionJob.document_id == document_id).delete()
        db.query(Document).filter(Document.id == document_id).delete()
        db.commit()

    monkeypatch.setattr(client_service, "delete_document", _fake_delete_document)

    client_service.ClientDeletionService(db_session).delete_client(target_client.id)

    deleted_doc_ids = {doc_id for doc_id, hard in deleted_docs if hard}
    assert deleted_doc_ids == {first_doc.id, second_doc.id}

    assert db_session.query(Client).filter(Client.id == target_client.id).count() == 0
    assert (
        db_session.query(Document)
        .filter(Document.client_id == target_client.id)
        .count()
        == 0
    )
    assert (
        db_session.query(VectorNodeRegistry)
        .filter(VectorNodeRegistry.client_id == target_client.id)
        .count()
        == 0
    )
    assert (
        db_session.query(IngestionJob)
        .filter(IngestionJob.client_id == target_client.id)
        .count()
        == 0
    )
    assert (
        db_session.query(QueryLog).filter(QueryLog.client_id == target_client.id).count()
        == 0
    )
    assert (
        db_session.query(RetrievalLog)
        .filter(RetrievalLog.query_log_id == "query-target")
        .count()
        == 0
    )
    assert (
        db_session.query(ConflictLog)
        .filter(ConflictLog.query_log_id == "query-target")
        .count()
        == 0
    )
    assert (
        db_session.query(UserClientAccess)
        .filter(UserClientAccess.client_id == target_client.id)
        .count()
        == 0
    )

    assert (
        db_session.query(Client).filter(Client.id == untouched_client.id).count() == 1
    )
    assert (
        db_session.query(Document)
        .filter(Document.client_id == untouched_client.id)
        .count()
        == 1
    )
    assert (
        db_session.query(QueryLog)
        .filter(QueryLog.client_id == untouched_client.id)
        .count()
        == 1
    )


def test_delete_client_raises_when_client_missing(db_session):
    with pytest.raises(ValueError, match="Client missing-client not found."):
        client_service.ClientDeletionService(db_session).delete_client("missing-client")
