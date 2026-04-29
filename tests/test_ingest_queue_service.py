from pathlib import Path

import pytest

from app.db.models import Document, IngestionJob
from app.services import ingest_service


def test_enqueue_document_ingestion_creates_queued_records(
    db_session, seeded_entities, monkeypatch, tmp_path: Path
):
    client = seeded_entities["client"]
    captured_tasks = []

    class FakeQueueManager:
        def enqueue(self, task):
            captured_tasks.append(task)

    def fake_save_upload_file(file_content: bytes, filename: str, _dest_folder: str) -> str:
        path = tmp_path / filename
        path.write_bytes(file_content)
        return str(path)

    monkeypatch.setattr(ingest_service, "save_upload_file", fake_save_upload_file)
    monkeypatch.setattr(
        ingest_service,
        "get_ingestion_queue_manager",
        lambda: FakeQueueManager(),
    )

    doc, job = ingest_service.enqueue_document_ingestion(
        file_content=b"sample pdf bytes",
        filename="queued.pdf",
        client_id=client.id,
        client_name=client.name,
        db=db_session,
    )

    assert doc.status == "queued"
    assert job.status == "queued"
    assert Path(doc.storage_path).exists()
    assert len(captured_tasks) == 1
    assert captured_tasks[0].document_id == doc.id
    assert captured_tasks[0].job_id == job.id

    persisted_doc = db_session.query(Document).filter(Document.id == doc.id).one()
    persisted_job = db_session.query(IngestionJob).filter(IngestionJob.id == job.id).one()
    assert persisted_doc.status == "queued"
    assert persisted_job.status == "queued"


def test_enqueue_document_ingestion_marks_failure_when_queue_rejects(
    db_session, seeded_entities, monkeypatch, tmp_path: Path
):
    client = seeded_entities["client"]

    class RejectingQueueManager:
        def enqueue(self, _task):
            raise ValueError("queue full")

    def fake_save_upload_file(file_content: bytes, filename: str, _dest_folder: str) -> str:
        path = tmp_path / filename
        path.write_bytes(file_content)
        return str(path)

    monkeypatch.setattr(ingest_service, "save_upload_file", fake_save_upload_file)
    monkeypatch.setattr(
        ingest_service,
        "get_ingestion_queue_manager",
        lambda: RejectingQueueManager(),
    )

    with pytest.raises(ValueError, match="Failed to queue ingestion"):
        ingest_service.enqueue_document_ingestion(
            file_content=b"sample pdf bytes",
            filename="queue_fail.pdf",
            client_id=client.id,
            client_name=client.name,
            db=db_session,
        )

    failed_doc = db_session.query(Document).filter(Document.name == "queue_fail.pdf").one()
    failed_job = (
        db_session.query(IngestionJob).filter(IngestionJob.document_id == failed_doc.id).one()
    )
    assert failed_doc.status == "failed"
    assert failed_job.status == "failed"
    assert "queue full" in (failed_job.error_message or "").lower()
