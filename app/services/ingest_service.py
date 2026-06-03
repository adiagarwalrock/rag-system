"""
Ingest service: orchestrates the full document ingestion pipeline.

Flow: validate -> save -> parse -> version resolve -> index -> persist mappings/status
"""

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Full, Queue
from threading import Lock, Thread
from typing import Any, List

from llama_index.core import Settings as LlamaSettings
from llama_index.core.extractors import (
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo, TextNode
from sqlalchemy import true
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models.client import Client
from app.db.models.document import (
    Document,
    DocumentVersion,
    IngestionJob,
    VectorNodeRegistry,
)
from app.db.snowflake import SessionLocal
from app.indexing.vector_store import COLLECTION_NAME, vector_store_manager
from app.ingestion.parser import parse_document, save_upload_file
from app.ingestion.parser.registry import PARSER_AUTO, VALID_PARSERS
from app.ingestion.validator import (
    compute_checksum,
    validate_file_size,
    validate_file_type,
)
from app.ingestion.version_resolver import resolve_version
from app.ingestion.retrieval_metadata import normalize_retrieval_metadata

logger = logging.getLogger(__name__)

RAW_DATA_DIR = settings.RAW_DATA_DIR

NON_SEMANTIC_EMBED_METADATA_KEYS = (
    "document_id",
    "doc_id",
    "ref_doc_id",
    "client_id",
    "client_name",
    "ingestion_job_id",
    "parser_name",
    "parser_version",
    "file_name",
    "source_file",
    "file_type",
    "chunk_id",
    "chunk_type",
    "citation_label",
    "version_label",
    "document_version_group",
    "version_rank",
    "published_at",
    "effective_from",
    "effective_to",
    "is_current",
    "authority_score",
    "table_detected",
    "chart_detected",
    "contains_numeric_data",
    "page_num",
    "page_nums",
    "slide_num",
    "section_path",
    "region_ids",
    "source_artifact_type",
    "source_artifact_id",
    "bbox_refs",
    "caption",
    "numeric_density",
    "layout_confidence",
    "complexity_score",
    "asset_refs",
    "parser_sources",
    "page_class",
    "artifact_bundle_path",
    "table_title",
    "figure_type",
    "chart_type",
    "chart_title",
    "x_axis_label",
    "y_axis_label",
    "x_categories",
    "series",
    "approx_datapoints",
    "trend_summary",
    "key_chart_facts",
    "numeric_extraction_confidence",
    "chart_parse_status",
    "llm_caption_model",
    "llm_caption_version",
    "llm_caption_prompt_version",
    "llm_caption_status",
    "llm_caption_error",
    "units",
    "continuation_flag",
    "ocr_used",
    "table_id",
    "reasoning_type",
    "source_artifact_ids",
    "evidence_refs",
    "reasoning_confidence",
    "reasoning_model",
    "reasoning_prompt_version",
    "claims",
    "llm_enriched",
    # temporal / scope anchors — stay in LLM metadata, excluded from embeddings
    "document_date",
    "as_of_date",
    "metric_basis",
    "document_type",
)

NON_SEMANTIC_LLM_METADATA_KEYS = (
    "document_id",
    "doc_id",
    "ref_doc_id",
    "client_id",
    "client_name",
    "ingestion_job_id",
    "parser_name",
    "parser_version",
    "file_type",
    "chunk_id",
    "chunk_type",
    "citation_label",
    "version_rank",
    "published_at",
    "is_current",
    "page_num",
    "page_nums",
    "source_artifact_type",
    "source_artifact_id",
    "chart_type",
    "chart_title",
    "chart_parse_status",
    "llm_caption_status",
    "asset_refs",
    "artifact_bundle_path",
    "reasoning_type",
    "source_artifact_ids",
    "evidence_refs",
    "reasoning_confidence",
    "reasoning_model",
    "reasoning_prompt_version",
    "claims",
    "llm_enriched",
)

_MISSING_DOC_ID_SENTINELS = {"", "none", "null", "n/a", "na", "undefined"}
_LEGACY_JOB_PARSER_NAME = "rag_ingestion_pipeline"


@dataclass(frozen=True, slots=True)
class IngestionQueueTask:
    document_id: str
    job_id: str
    client_id: str
    client_name: str
    parser_preference: str | None = None


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
            if not file_path or not os.path.exists(file_path):
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
                    parser_preference=task.parser_preference,
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


@dataclass(frozen=True, slots=True)
class IngestionExecutionContext:
    db_doc: Document
    job: IngestionJob
    file_path: str
    filename: str
    client_id: str
    client_name: str
    doc_id: str
    file_ext: str
    db: Session
    parser_preference: str | None = None


_INGESTION_QUEUE_MANAGER: IngestionQueueManager | None = None
_INGESTION_QUEUE_LOCK = Lock()


def get_ingestion_queue_manager() -> IngestionQueueManager:
    global _INGESTION_QUEUE_MANAGER
    if _INGESTION_QUEUE_MANAGER is not None:
        return _INGESTION_QUEUE_MANAGER

    with _INGESTION_QUEUE_LOCK:
        if _INGESTION_QUEUE_MANAGER is None:
            _INGESTION_QUEUE_MANAGER = IngestionQueueManager(
                max_workers=settings.INGESTION_MAX_WORKERS,
                max_queue_size=settings.INGESTION_QUEUE_MAX_SIZE,
            )

    return _INGESTION_QUEUE_MANAGER


def _build_non_layout_node_parser() -> Any:
    embed_model: Any | None = getattr(LlamaSettings, "_embed_model", None)
    if embed_model is None and not settings.is_openai_api_key_placeholder:
        vector_store_manager.configure_llama_settings()
        embed_model = getattr(LlamaSettings, "_embed_model", None)

    return SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        breakpoint_percentile_threshold=settings.SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE,
        buffer_size=settings.SEMANTIC_SPLITTER_BUFFER_SIZE,
    )


def _apply_metadata_exclusions(nodes: List[BaseNode]) -> None:
    for node in nodes:
        embed_excluded = set(node.excluded_embed_metadata_keys or [])
        llm_excluded = set(node.excluded_llm_metadata_keys or [])

        embed_excluded.update(NON_SEMANTIC_EMBED_METADATA_KEYS)
        llm_excluded.update(NON_SEMANTIC_LLM_METADATA_KEYS)

        node.excluded_embed_metadata_keys = sorted(embed_excluded)
        node.excluded_llm_metadata_keys = sorted(llm_excluded)


def _normalize_document_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in _MISSING_DOC_ID_SENTINELS:
            return None
        return normalized
    return str(value)


def _apply_ref_doc_ids(nodes: List[BaseNode]) -> None:
    """Ensure top-level vector-store doc IDs are derived from canonical metadata."""
    for node in nodes:
        metadata = node.metadata or {}
        document_id = _normalize_document_id(metadata.get("document_id"))
        if document_id:
            relationships = dict(node.relationships or {})
            relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=document_id
            )
            node.relationships = relationships


def _isoformat_or_none(value: Any) -> str | None:
    return value.isoformat() if value else None


def _core_version_metadata(version_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "version_rank": version_info.get("version_rank", 0),
        "published_at": _isoformat_or_none(version_info.get("published_at")),
        "is_current": version_info.get("is_current", False),
    }


def _document_version_metadata(version_info: dict[str, Any]) -> dict[str, Any]:
    return {
        **_core_version_metadata(version_info),
        "version_label": version_info.get("version_label"),
        "document_version_group": version_info.get("version_group"),
        "effective_from": _isoformat_or_none(version_info.get("effective_from")),
        "effective_to": _isoformat_or_none(version_info.get("effective_to")),
        "document_type": version_info.get("document_type"),
    }


def _apply_retrieval_metadata(
    nodes: List[BaseNode],
    filename: str,
    version_info: dict[str, Any],
) -> None:
    for index, node in enumerate(nodes, start=1):
        metadata = node.metadata or {}
        chunk_id = metadata.get("chunk_id") or str(uuid.uuid4())
        page_num = metadata.get("page_num")
        slide_num = metadata.get("slide_num")

        label_parts = [filename]
        if page_num:
            label_parts.append(f"p.{page_num}")
        if slide_num:
            label_parts.append(f"slide {slide_num}")
        label_parts.append(f"chunk {index}")

        metadata.update(
            {
                "chunk_id": chunk_id,
                "source_file": metadata.get("source_file") or filename,
                "citation_label": metadata.get("citation_label")
                or " - ".join(label_parts),
            }
        )
        metadata = normalize_retrieval_metadata(
            metadata,
            text=getattr(node, "text", "") or "",
            version_info=version_info,
            source_file=filename,
            chunk_index=index,
        )
        node.metadata = metadata


def _create_document_record(
    *,
    doc_id: str,
    client_id: str,
    filename: str,
    file_ext: str,
    checksum: str,
    status: str,
) -> Document:
    return Document(
        id=doc_id,
        client_id=client_id,
        name=filename,
        file_type=file_ext,
        storage_path="",
        checksum=checksum,
        status=status,
    )


def _create_ingestion_job_record(
    *,
    job_id: str,
    client_id: str,
    doc_id: str,
    status: str,
    file_size: int,
    started_at: datetime | None,
    parser_preference: str | None = None,
) -> IngestionJob:
    return IngestionJob(
        id=job_id,
        client_id=client_id,
        document_id=doc_id,
        status=status,
        started_at=started_at,
        parser_name=_normalize_parser_intent(parser_preference) or PARSER_AUTO,
        parser_version=None,
        filesize_bytes=file_size,
    )


def _normalize_parser_intent(parser_preference: str | None) -> str | None:
    parser = (parser_preference or "").strip().lower()
    if not parser:
        return PARSER_AUTO
    if parser == _LEGACY_JOB_PARSER_NAME:
        return None
    if parser not in VALID_PARSERS:
        return None
    return parser


def _retry_parser_preference(db: Session, document_id: str) -> str | None:
    jobs = (
        db.query(IngestionJob)
        .filter(IngestionJob.document_id == document_id)
        .order_by(IngestionJob.started_at.desc(), IngestionJob.id.desc())
        .all()
    )
    for job in jobs:
        parser = _normalize_parser_intent(job.parser_name)
        if parser is None:
            continue
        return None if parser == PARSER_AUTO else parser
    return None


def ingest_document(
    file_content: bytes,
    filename: str,
    client_id: str,
    client_name: str,
    db: Session,
    parser_preference: str | None = None,
) -> Document:
    """
    Full ingestion pipeline for a single document using LlamaIndex IngestionPipeline.

    Returns the persisted Document record.
    """
    # 1. Validate
    file_ext = validate_file_type(filename)
    file_size = validate_file_size(file_content)
    checksum = compute_checksum(file_content)

    # 2. Create document record
    doc_id = str(uuid.uuid4())
    db_doc = _create_document_record(
        doc_id=doc_id,
        client_id=client_id,
        filename=filename,
        file_ext=file_ext,
        checksum=checksum,
        status="processing",
    )
    db.add(db_doc)

    # 3. Create ingestion job
    job_id = str(uuid.uuid4())
    job = _create_ingestion_job_record(
        job_id=job_id,
        client_id=client_id,
        doc_id=doc_id,
        status="running",
        file_size=file_size,
        started_at=datetime.now(timezone.utc),
        parser_preference=parser_preference,
    )
    db.add(job)
    db.commit()

    try:
        # 4. Save file
        file_path = save_upload_file(file_content, filename, RAW_DATA_DIR)
        db_doc.storage_path = file_path

        _execute_pipeline(
            db_doc,
            job,
            file_path,
            filename,
            client_id,
            client_name,
            doc_id,
            file_ext,
            db,
            parser_preference=parser_preference,
        )
        return db_doc
    except Exception as e:
        _handle_ingestion_failure(e, filename, doc_id, job_id, db_doc, job, db)
        raise


def enqueue_document_ingestion(
    file_content: bytes,
    filename: str,
    client_id: str,
    client_name: str,
    db: Session,
    parser_preference: str | None = None,
) -> tuple[Document, IngestionJob]:
    """
    Queue document ingestion for background processing.

    Returns:
        Tuple of persisted (Document, IngestionJob) in queued/running lifecycle.
    """
    file_ext = validate_file_type(filename)
    file_size = validate_file_size(file_content)
    checksum = compute_checksum(file_content)

    doc_id = str(uuid.uuid4())
    db_doc = _create_document_record(
        doc_id=doc_id,
        client_id=client_id,
        filename=filename,
        file_ext=file_ext,
        checksum=checksum,
        status="queued",
    )
    db.add(db_doc)

    job_id = str(uuid.uuid4())
    job = _create_ingestion_job_record(
        job_id=job_id,
        client_id=client_id,
        doc_id=doc_id,
        status="queued",
        file_size=file_size,
        started_at=None,
        parser_preference=parser_preference,
    )
    db.add(job)
    db.commit()

    try:
        file_path = save_upload_file(file_content, filename, RAW_DATA_DIR)
        db_doc.storage_path = file_path
        db.commit()
    except Exception as exc:
        _handle_ingestion_failure(exc, filename, doc_id, job_id, db_doc, job, db)
        raise

    task = IngestionQueueTask(
        document_id=doc_id,
        job_id=job_id,
        client_id=client_id,
        client_name=client_name,
        parser_preference=parser_preference,
    )
    try:
        get_ingestion_queue_manager().enqueue(task)
    except Exception as exc:
        _handle_ingestion_failure(exc, filename, doc_id, job_id, db_doc, job, db)
        raise ValueError(f"Failed to queue ingestion: {exc}") from exc

    db.refresh(db_doc)
    db.refresh(job)
    return db_doc, job


def retry_ingestion(document_id: str, db: Session) -> Document:
    """
    Retry ingestion for a failed document.
    """
    # 1. Fetch document and validate
    db_doc = db.query(Document).filter(Document.id == document_id).first()
    if not db_doc:
        raise ValueError(f"Document {document_id} not found.")

    if db_doc.status not in ["failed", "indexed", "completed"]:
        raise ValueError(f"Cannot retry document in '{db_doc.status}' state.")

    if not db_doc.storage_path or not os.path.exists(db_doc.storage_path):
        raise ValueError("Raw file is missing. Please re-upload the document.")

    client_record = db.query(Client).filter(Client.id == db_doc.client_id).first()
    client_name = client_record.name if client_record else "Unknown"

    file_path = db_doc.storage_path
    filename = db_doc.name
    client_id = db_doc.client_id
    doc_id = db_doc.id
    file_ext = db_doc.file_type
    retry_parser_preference = _retry_parser_preference(db, doc_id)

    # 3. Pre-clean prior partial data
    try:
        vector_store_manager.delete_document_vectors(doc_id, client_id)
        db.query(VectorNodeRegistry).filter(
            VectorNodeRegistry.document_id == doc_id
        ).delete()
        db.query(DocumentVersion).filter(DocumentVersion.document_id == doc_id).delete()
        db.commit()
    except Exception as e:
        db.rollback()
        raise ValueError(f"Failed to pre-clean prior data: {e}")

    # 2. Mark running, create IngestionJob
    db_doc.status = "processing"
    job_id = str(uuid.uuid4())
    job = _create_ingestion_job_record(
        job_id=job_id,
        client_id=client_id,
        doc_id=doc_id,
        status="running",
        file_size=os.path.getsize(file_path),
        started_at=datetime.now(timezone.utc),
        parser_preference=retry_parser_preference,
    )
    db.add(job)
    db.commit()

    try:
        _execute_pipeline(
            db_doc,
            job,
            file_path,
            filename,
            client_id,
            client_name,
            doc_id,
            file_ext,
            db,
            parser_preference=retry_parser_preference,
        )
        return db_doc
    except Exception as e:
        _handle_ingestion_failure(e, filename, doc_id, job_id, db_doc, job, db)
        raise


def delete_document(document_id: str, db: Session, hard: bool = False) -> None:
    db_doc = db.query(Document).filter(Document.id == document_id).first()
    if not db_doc:
        raise ValueError(f"Document {document_id} not found.")

    client_id = db_doc.client_id
    db_doc.status = "deleting"
    db.commit()

    try:
        # 2. delete vector nodes
        vector_deleted = vector_store_manager.delete_document_vectors(
            document_id, client_id
        )
        if not vector_deleted:
            logger.warning(f"Failed to delete vectors for document {document_id}")

        # 3. delete db rows
        db.query(VectorNodeRegistry).filter(
            VectorNodeRegistry.document_id == document_id
        ).delete()
        db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id
        ).delete()
        db.query(IngestionJob).filter(IngestionJob.document_id == document_id).delete()

        # 4. remove raw file
        if db_doc.storage_path and os.path.exists(db_doc.storage_path):
            try:
                os.remove(db_doc.storage_path)
            except OSError as e:
                logger.warning(f"Failed to remove file {db_doc.storage_path}: {e}")

        # 5. set status deleted or hard delete
        if hard:
            db.delete(db_doc)
        else:
            db_doc.status = "deleted"
            db_doc.storage_path = ""

        db.commit()
    except Exception as e:
        db.rollback()
        db_doc.status = "deleting_failed"
        db.commit()
        raise ValueError(f"Failed to delete document: {e}")


class IngestionPipelineExecutor:
    """Coordinates parse -> enrich -> index -> persist for one document."""

    def __init__(self, context: IngestionExecutionContext):
        self.context = context
        self.db_doc = context.db_doc
        self.job = context.job
        self.file_path = context.file_path
        self.filename = context.filename
        self.client_id = context.client_id
        self.client_name = context.client_name
        self.doc_id = context.doc_id
        self.file_ext = context.file_ext
        self.db = context.db

    def run(self) -> tuple[int, int]:
        vector_store_manager.configure_llama_settings()

        document_metadata = self._build_document_metadata()
        llama_docs, units = parse_document(
            self.file_path,
            document_metadata,
            parser_preference=self.context.parser_preference,
        )
        self._apply_parser_metadata(llama_docs)

        version_info = self._resolve_version_info(llama_docs)
        self._persist_version_record(version_info)
        self._apply_document_metadata(
            llama_docs=llama_docs,
            units=units,
            document_metadata=document_metadata,
            version_info=version_info,
        )

        nodes = self._run_ingestion_pipeline(llama_docs=llama_docs)

        _apply_retrieval_metadata(
            nodes, filename=self.filename, version_info=version_info
        )
        _apply_ref_doc_ids(nodes)
        _apply_metadata_exclusions(nodes)

        self._index_nodes(nodes)
        self._persist_registry_rows(nodes)
        self._mark_success()
        return len(units), len(nodes)

    def _build_document_metadata(self) -> dict[str, Any]:
        return {
            "document_id": self.doc_id,
            "client_id": self.client_id,
            "client_name": self.client_name,
            "document_name": self.filename,
            "file_name": self.filename,
            "file_type": self.file_ext,
            "ingestion_job_id": self.job.id,
        }

    def _apply_parser_metadata(self, llama_docs: List[Any]) -> None:
        if not llama_docs:
            return
        parser_name = llama_docs[0].metadata.get("parser_name")
        parser_version = llama_docs[0].metadata.get("parser_version")
        if parser_name:
            self.job.parser_name = parser_name
        if parser_version:
            self.job.parser_version = parser_version

    def _resolve_version_info(self, llama_docs: List[Any]) -> dict[str, Any]:
        content_preview = llama_docs[0].text[:500] if llama_docs else ""
        return resolve_version(self.filename, content_preview)

    def _persist_version_record(self, version_info: dict[str, Any]) -> None:
        version_record = DocumentVersion(
            id=str(uuid.uuid4()),
            document_id=self.doc_id,
            version_label=version_info.get("version_label"),
            version_group=version_info.get("version_group"),
            version_rank=version_info.get("version_rank", 0),
            published_at=version_info.get("published_at"),
            effective_from=version_info.get("effective_from"),
            effective_to=version_info.get("effective_to"),
            is_current=version_info.get("is_current", False),
            confidence_score=version_info.get("confidence_score", 0.0),
        )
        self.db.add(version_record)

        version_group = version_info.get("version_group")
        if not version_group:
            return

        self.db_doc.document_family = version_group
        if version_info.get("is_current"):
            _supersede_older_versions(
                self.db,
                self.client_id,
                self.doc_id,
                version_group,
            )

    def _apply_document_metadata(
        self,
        *,
        llama_docs: List[Any],
        units: list[dict[str, Any]],
        document_metadata: dict[str, Any],
        version_info: dict[str, Any],
    ) -> None:
        for index, doc in enumerate(llama_docs):
            unit = units[index] if index < len(units) else {}
            doc.metadata.update(
                {
                    **document_metadata,
                    "version_label": version_info.get("version_label"),
                    "document_version_group": version_info.get("version_group"),
                    "effective_from": _isoformat_or_none(
                        version_info.get("effective_from")
                    ),
                    "effective_to": _isoformat_or_none(
                        version_info.get("effective_to")
                    ),
                    "table_detected": unit.get("table_detected", False),
                    "chart_detected": unit.get("chart_detected", False),
                    "contains_numeric_data": unit.get("contains_numeric_data", False),
                    "chunk_type": unit.get(
                        "chunk_type", doc.metadata.get("chunk_type", "text")
                    ),
                    "page_nums": unit.get("page_nums", doc.metadata.get("page_nums")),
                    "section_path": unit.get(
                        "section_path", doc.metadata.get("section_path")
                    ),
                    "source_artifact_type": unit.get(
                        "source_artifact_type",
                        doc.metadata.get("source_artifact_type"),
                    ),
                    "source_artifact_id": unit.get(
                        "source_artifact_id",
                        doc.metadata.get("source_artifact_id"),
                    ),
                    "layout_confidence": unit.get(
                        "layout_confidence", doc.metadata.get("layout_confidence")
                    ),
                    "complexity_score": unit.get(
                        "complexity_score", doc.metadata.get("complexity_score")
                    ),
                    "artifact_bundle_path": unit.get(
                        "artifact_bundle_path",
                        doc.metadata.get("artifact_bundle_path"),
                    ),
                    "authority_score": 1.0,
                    # temporal / scope anchors populated by external parser Extract API
                    "document_date": unit.get(
                        "document_date", doc.metadata.get("document_date")
                    ),
                    "as_of_date": unit.get(
                        "as_of_date", doc.metadata.get("as_of_date")
                    ),
                    "metric_basis": unit.get(
                        "metric_basis", doc.metadata.get("metric_basis")
                    ),
                }
            )
            doc.metadata = normalize_retrieval_metadata(
                doc.metadata,
                text=doc.text or "",
                document_metadata=document_metadata,
                version_info=version_info,
                source_file=self.filename,
                chunk_index=index + 1,
            )

    def _run_ingestion_pipeline(self, *, llama_docs: List[Any]) -> List[BaseNode]:
        # External parsers (Reducto, LlamaParse) and the layout-aware PDF pipeline
        # already produce intentionally-chunked LlamaDocuments — each doc is a
        # typed chunk (full_table, body_text, figure_artifact …) with metadata
        # bound to that exact text span.  Running SemanticSplitterNodeParser on
        # pre-chunked docs re-slices the text on semantic boundaries, splits
        # tables across nodes, and strips the chunk_type metadata association.
        # Detect pre-chunked docs by the presence of "chunk_type" in metadata and
        # skip the splitter for those, converting each doc directly to a TextNode.
        pre_chunked = [
            doc
            for doc in llama_docs
            if doc.metadata.get("chunk_type")
            and doc.metadata.get("chunk_type") != "text"
        ]
        needs_splitting = [
            doc
            for doc in llama_docs
            if not doc.metadata.get("chunk_type")
            or doc.metadata.get("chunk_type") == "text"
        ]

        nodes: List[BaseNode] = []

        # Build LLM extractors once — shared by both paths.
        llm_extractors: list[Any] = []
        try:
            llm_extractors = [
                TitleExtractor(nodes=5),
                SummaryExtractor(summaries=["prev", "self"]),
                KeywordExtractor(keywords=10),
                QuestionsAnsweredExtractor(num_questions=3),
            ]
            logger.info(
                "Initialized LLM-based extractors (Title, Summary, Keyword, Questions)"
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize LLM extractors: %s. Skipping.", exc
            )

        # ── Path A: pre-chunked docs — skip splitter, still run LLM extractors ──
        if pre_chunked:
            logger.info(
                "Bypassing semantic splitter for %d pre-chunked docs (%s)",
                len(pre_chunked),
                self.filename,
            )
            pre_chunked_nodes: List[BaseNode] = []
            for doc in pre_chunked:
                node = TextNode(
                    text=doc.text,
                    metadata=dict(doc.metadata),
                    excluded_embed_metadata_keys=list(
                        doc.excluded_embed_metadata_keys or []
                    ),
                    excluded_llm_metadata_keys=list(
                        doc.excluded_llm_metadata_keys or []
                    ),
                )
                pre_chunked_nodes.append(node)

            if llm_extractors:
                pipeline = IngestionPipeline(transformations=llm_extractors)
                # num_workers=1 keeps all nodes in a single sequential batch so
                # SummaryExtractor can compute prev_section_summary for every node
                # (multi-worker splits nodes across batches, breaking the i-1 window).
                pre_chunked_nodes = pipeline.run(
                    nodes=pre_chunked_nodes,
                    num_workers=1,
                )
            nodes.extend(pre_chunked_nodes)

        # ── Path B: legacy / un-chunked docs — run semantic splitter + extractors ─
        if needs_splitting:
            logger.info(
                "Using semantic splitter for %d docs (%s)",
                len(needs_splitting),
                self.filename,
            )
            transformations: list[Any] = [_build_non_layout_node_parser()]
            transformations.extend(llm_extractors)
            pipeline = IngestionPipeline(transformations=transformations)
            split_nodes = pipeline.run(documents=needs_splitting, num_workers=4)
            nodes.extend(split_nodes)

        return nodes

    def _index_nodes(self, nodes: List[BaseNode]) -> None:
        vector_store_manager.index_nodes(nodes)
        qdrant_points = vector_store_manager.get_qdrant_point_count()
        if qdrant_points is None:
            return
        logger.info(
            "Qdrant collection '%s' current point count: %d",
            COLLECTION_NAME,
            qdrant_points,
        )

    def _persist_registry_rows(self, nodes: List[BaseNode]) -> None:
        for node in nodes:
            self.db.add(
                VectorNodeRegistry(
                    id=str(uuid.uuid4()),
                    document_id=self.doc_id,
                    client_id=self.client_id,
                    vector_collection=COLLECTION_NAME,
                    vector_node_id=node.node_id,
                    embedding_model=settings.EMBEDDING_MODEL,
                )
            )

    def _mark_success(self) -> None:
        self.db_doc.status = "indexed"
        self.job.status = "completed"
        self.job.finished_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(self.db_doc)


def _execute_pipeline(
    db_doc: Document,
    job: IngestionJob,
    file_path: str,
    filename: str,
    client_id: str,
    client_name: str,
    doc_id: str,
    file_ext: str,
    db: Session,
    parser_preference: str | None = None,
):
    context = IngestionExecutionContext(
        db_doc=db_doc,
        job=job,
        file_path=file_path,
        filename=filename,
        client_id=client_id,
        client_name=client_name,
        doc_id=doc_id,
        file_ext=file_ext,
        db=db,
        parser_preference=parser_preference,
    )
    parsed_units, vector_rows = IngestionPipelineExecutor(context).run()
    logger.info(
        "Ingestion complete: doc=%s, parsed_units=%d, vector_registry_rows=%d",
        doc_id,
        parsed_units,
        vector_rows,
    )


def _handle_ingestion_failure(
    e: Exception,
    filename: str,
    doc_id: str,
    job_id: str,
    db_doc: Document,
    job: IngestionJob,
    db: Session,
):
    error_msg = str(e)[:1000]
    error_lower = error_msg.lower()
    if (
        "openai" in error_lower
        or "genai" in error_lower
        or "api key" in error_lower
        or "rate limit" in error_lower
    ):
        logger.exception("Embedding/LLM failure for %s: %s", filename, error_msg)
    elif (
        "qdrant" in error_lower
        or "connection" in error_lower
        or "vector" in error_lower
    ):
        logger.exception("Vector store failure for %s: %s", filename, error_msg)
    else:
        logger.exception("Ingestion failed for %s: %s", filename, error_msg)

    try:
        db.rollback()
        db_doc.status = "failed"
        job.status = "failed"
        job.error_message = error_msg
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception(
            "Failed to persist ingestion failure state for %s (doc=%s, job=%s)",
            filename,
            doc_id,
            job_id,
        )


def _supersede_older_versions(db: Session, client_id: str, doc_id: str, group: str):
    """Mark older versions in the same family as non-current."""
    older_versions = (
        db.query(DocumentVersion)
        .join(Document, Document.id == DocumentVersion.document_id)
        .filter(
            Document.client_id == client_id,
            Document.document_family == group,
            DocumentVersion.document_id != doc_id,
            DocumentVersion.is_current == true(),
        )
        .all()
    )
    for old_ver in older_versions:
        old_ver.is_current = False
        logger.info(
            "Marked version %s (doc %s) as non-current, superseded by %s",
            old_ver.version_label,
            old_ver.document_id,
            doc_id,
        )
