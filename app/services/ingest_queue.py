import logging
from dataclasses import dataclass
from queue import Full, Queue
from threading import Lock, Thread

from app.db.models.document import Document, IngestionJob
from app.db.snowflake import SessionLocal

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionQueueTask:
    document_id: str
    job_id: str
    client_id: str
    client_name: str


class IngestionQueueManager:
    """Background ingestion queue with bounded worker concurrency."""

    def __init__(self, max_workers: int, max_queue_size: int):
        self._max_workers = max(1, max_workers)
        self._queue: Queue[IngestionQueueTask] = Queue(maxsize=max_queue_size)
        self._workers_started = False
        self._lock = Lock()

    def enqueue(self, task: IngestionQueueTask) -> None:
        self._ensure_workers_started()
        try:
            self._queue.put_nowait(task)
        except Full as exc:
            raise ValueError(
                "Ingestion queue is full. Please retry in a moment."
            ) from exc

        logger.info(
            "Queued ingestion task doc=%s job=%s queue_size=%d",
            task.document_id,
            task.job_id,
            self._queue.qsize(),
        )

    def _ensure_workers_started(self) -> None:
        if self._workers_started:
            return

        with self._lock:
            if self._workers_started:
                return

            for worker_index in range(self._max_workers):
                worker = Thread(
                    target=self._worker_loop,
                    args=(worker_index,),
                    daemon=True,
                    name=f"ingestion-worker-{worker_index + 1}",
                )
                worker.start()
            self._workers_started = True
            logger.info(
                "Started ingestion queue workers: count=%d",
                self._max_workers,
            )

    def _worker_loop(self, worker_index: int) -> None:
        while True:
            task = self._queue.get()
            try:
                self._run_task(task, worker_index=worker_index)
            except Exception:
                logger.exception(
                    "Unhandled ingestion worker failure for doc=%s",
                    task.document_id,
                )
            finally:
                self._queue.task_done()

    def _run_task(self, task: IngestionQueueTask, worker_index: int) -> None:
        from datetime import datetime, timezone
        from pathlib import Path

        from app.services.ingest_service import (
            _execute_pipeline,
            _handle_ingestion_failure,
        )

        with SessionLocal() as db:
            db_doc = db.query(Document).filter(Document.id == task.document_id).first()
            job = db.query(IngestionJob).filter(IngestionJob.id == task.job_id).first()

            if not db_doc or not job:
                logger.warning(
                    "Skipping queued task for missing entities doc=%s job=%s",
                    task.document_id,
                    task.job_id,
                )
                return

            if db_doc.status in {"deleted", "deleting", "deleting_failed"}:
                logger.info(
                    "Skipping queued ingestion for document in terminal delete state: %s",
                    db_doc.id,
                )
                return

            file_path = db_doc.storage_path
            if not file_path or not Path(file_path).exists():
                missing_error = ValueError(
                    "Raw file is missing before queued ingestion started."
                )
                _handle_ingestion_failure(
                    missing_error,
                    db_doc.name,
                    db_doc.id,
                    job.id,
                    db_doc,
                    job,
                    db,
                )
                return

            db_doc.status = "processing"
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            db.commit()

            logger.info(
                "Worker %d started ingestion doc=%s job=%s",
                worker_index + 1,
                db_doc.id,
                job.id,
            )

            try:
                _execute_pipeline(
                    db_doc,
                    job,
                    file_path,
                    db_doc.name,
                    task.client_id,
                    task.client_name,
                    db_doc.id,
                    db_doc.file_type,
                    db,
                )
            except Exception as exc:
                _handle_ingestion_failure(
                    exc,
                    db_doc.name,
                    db_doc.id,
                    job.id,
                    db_doc,
                    job,
                    db,
                )


_INGESTION_QUEUE_MANAGER: IngestionQueueManager | None = None
_INGESTION_QUEUE_LOCK = Lock()


def get_ingestion_queue_manager() -> IngestionQueueManager:
    global _INGESTION_QUEUE_MANAGER
    if _INGESTION_QUEUE_MANAGER is not None:
        return _INGESTION_QUEUE_MANAGER

    with _INGESTION_QUEUE_LOCK:
        if _INGESTION_QUEUE_MANAGER is None:
            # We access settings here to avoid circular imports if any
            from app.core.config import settings

            _INGESTION_QUEUE_MANAGER = IngestionQueueManager(
                max_workers=settings.INGESTION_MAX_WORKERS,
                max_queue_size=settings.INGESTION_QUEUE_MAX_SIZE,
            )

    return _INGESTION_QUEUE_MANAGER
