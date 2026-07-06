from pathlib import Path
from datetime import datetime, timezone

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
        parser_preference="llamaparse",
    )

    assert doc.status == "queued"
    assert job.status == "queued"
    assert Path(doc.storage_path).exists()
    assert len(captured_tasks) == 1
    assert captured_tasks[0].document_id == doc.id
    assert captured_tasks[0].job_id == job.id
    assert captured_tasks[0].parser_preference == "llamaparse"

    persisted_doc = db_session.query(Document).filter(Document.id == doc.id).one()
    persisted_job = db_session.query(IngestionJob).filter(IngestionJob.id == job.id).one()
    assert persisted_doc.status == "queued"
    assert persisted_job.status == "queued"
    assert persisted_job.parser_name == "llamaparse"


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


def test_retry_ingestion_reuses_latest_named_parser(
    db_session, seeded_entities, monkeypatch, tmp_path: Path
):
    document = seeded_entities["document"]
    document.status = "failed"
    document.storage_path = str(tmp_path / "policy_v1.pdf")
    Path(document.storage_path).write_bytes(b"sample pdf bytes")
    db_session.add(
        IngestionJob(
            id="failed-job",
            client_id=document.client_id,
            document_id=document.id,
            status="failed",
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            parser_name="llamaparse",
            parser_version=None,
            filesize_bytes=16,
        )
    )
    db_session.commit()
    captured = {}

    monkeypatch.setattr(
        ingest_service.vector_store_manager,
        "delete_document_vectors",
        lambda *_args, **_kwargs: True,
    )

    def fake_execute_pipeline(*_args, parser_preference=None, **_kwargs):
        captured["parser_preference"] = parser_preference
        document.status = "indexed"

    monkeypatch.setattr(ingest_service, "_execute_pipeline", fake_execute_pipeline)

    ingest_service.retry_ingestion(document.id, db_session)

    assert captured["parser_preference"] == "llamaparse"
    retry_job = (
        db_session.query(IngestionJob)
        .filter(IngestionJob.document_id == document.id, IngestionJob.status == "running")
        .one()
    )
    assert retry_job.parser_name == "llamaparse"


@pytest.mark.parametrize("parser_name", ["auto", "rag_ingestion_pipeline"])
def test_retry_ingestion_uses_auto_for_auto_or_legacy_placeholder(
    db_session, seeded_entities, monkeypatch, tmp_path: Path, parser_name: str
):
    document = seeded_entities["document"]
    document.status = "failed"
    document.storage_path = str(tmp_path / f"{parser_name}.pdf")
    Path(document.storage_path).write_bytes(b"sample pdf bytes")
    db_session.add(
        IngestionJob(
            id=f"failed-{parser_name}",
            client_id=document.client_id,
            document_id=document.id,
            status="failed",
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            parser_name=parser_name,
            parser_version=None,
            filesize_bytes=16,
        )
    )
    db_session.commit()
    captured = {}

    monkeypatch.setattr(
        ingest_service.vector_store_manager,
        "delete_document_vectors",
        lambda *_args, **_kwargs: True,
    )

    def fake_execute_pipeline(*_args, parser_preference=None, **_kwargs):
        captured["parser_preference"] = parser_preference
        document.status = "indexed"

    monkeypatch.setattr(ingest_service, "_execute_pipeline", fake_execute_pipeline)

    ingest_service.retry_ingestion(document.id, db_session)

    assert captured["parser_preference"] is None


def test_failed_retry_job_records_parser_intent(
    db_session, seeded_entities, monkeypatch, tmp_path: Path
):
    document = seeded_entities["document"]
    document.status = "failed"
    document.storage_path = str(tmp_path / "policy_failed_retry.pdf")
    Path(document.storage_path).write_bytes(b"sample pdf bytes")
    db_session.add(
        IngestionJob(
            id="failed-job",
            client_id=document.client_id,
            document_id=document.id,
            status="failed",
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            parser_name="reducto",
            parser_version=None,
            filesize_bytes=16,
        )
    )
    db_session.commit()

    monkeypatch.setattr(
        ingest_service.vector_store_manager,
        "delete_document_vectors",
        lambda *_args, **_kwargs: True,
    )

    def fail_execute_pipeline(*_args, **_kwargs):
        raise RuntimeError("parser failed again")

    monkeypatch.setattr(ingest_service, "_execute_pipeline", fail_execute_pipeline)

    with pytest.raises(RuntimeError, match="parser failed again"):
        ingest_service.retry_ingestion(document.id, db_session)

    retry_job = (
        db_session.query(IngestionJob)
        .filter(IngestionJob.document_id == document.id)
        .order_by(IngestionJob.started_at.desc(), IngestionJob.id.desc())
        .first()
    )
    assert retry_job.status == "failed"
    assert retry_job.parser_name == "reducto"
    assert "parser failed again" in retry_job.error_message
