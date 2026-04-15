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
    DocumentContextExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo
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
from app.ingestion.validator import (
    compute_checksum,
    validate_file_size,
    validate_file_type,
)
from app.ingestion.version_resolver import resolve_version

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
            _INGESTION_QUEUE_MANAGER = IngestionQueueManager(
                max_workers=settings.INGESTION_MAX_WORKERS,
                max_queue_size=settings.INGESTION_QUEUE_MAX_SIZE,
            )

    return _INGESTION_QUEUE_MANAGER


def _build_non_layout_node_parser() -> Any:
    embed_model = getattr(LlamaSettings, "_embed_model", None)
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
                **_core_version_metadata(version_info),
            }
        )
        node.metadata = metadata


def ingest_document(
    file_content: bytes,
    filename: str,
    client_id: str,
    client_name: str,
    db: Session,
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
    db_doc = Document(
        id=doc_id,
        client_id=client_id,
        name=filename,
        file_type=file_ext,
        storage_path="",
        checksum=checksum,
        status="processing",
    )
    db.add(db_doc)

    # 3. Create ingestion job
    job_id = str(uuid.uuid4())
    job = IngestionJob(
        id=job_id,
        client_id=client_id,
        document_id=doc_id,
        status="running",
        started_at=datetime.now(timezone.utc),
        parser_name="rag_ingestion_pipeline",
        parser_version="2.0.0",
        filesize_bytes=file_size,
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
    db_doc = Document(
        id=doc_id,
        client_id=client_id,
        name=filename,
        file_type=file_ext,
        storage_path="",
        checksum=checksum,
        status="queued",
    )
    db.add(db_doc)

    job_id = str(uuid.uuid4())
    job = IngestionJob(
        id=job_id,
        client_id=client_id,
        document_id=doc_id,
        status="queued",
        started_at=None,
        parser_name="rag_ingestion_pipeline",
        parser_version="2.0.0",
        filesize_bytes=file_size,
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
    job = IngestionJob(
        id=job_id,
        client_id=client_id,
        document_id=doc_id,
        status="running",
        started_at=datetime.now(timezone.utc),
        parser_name="rag_ingestion_pipeline",
        parser_version="2.0.0",
        filesize_bytes=os.path.getsize(file_path),
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
):
    vector_store_manager.configure_llama_settings()

    # 5. Parse
    document_metadata = {
        "document_id": doc_id,
        "client_id": client_id,
        "client_name": client_name,
        "document_name": filename,
        "file_name": filename,
        "file_type": file_ext,
        "ingestion_job_id": job.id,
    }
    llama_docs, units = parse_document(file_path, document_metadata)
    if llama_docs:
        parser_name = llama_docs[0].metadata.get("parser_name")
        parser_version = llama_docs[0].metadata.get("parser_version")
        if parser_name:
            job.parser_name = parser_name
        if parser_version:
            job.parser_version = parser_version

    # 6. Version resolution
    content_preview = ""
    if llama_docs:
        content_preview = llama_docs[0].text[:500]
    version_info = resolve_version(filename, content_preview)

    # Save version record
    version_record = DocumentVersion(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        version_label=version_info.get("version_label"),
        version_group=version_info.get("version_group"),
        version_rank=version_info.get("version_rank", 0),
        published_at=version_info.get("published_at"),
        effective_from=version_info.get("effective_from"),
        effective_to=version_info.get("effective_to"),
        is_current=version_info.get("is_current", False),
        confidence_score=version_info.get("confidence_score", 0.0),
    )
    db.add(version_record)

    # Also store document family and handle version supersession
    if version_info.get("version_group"):
        db_doc.document_family = version_info["version_group"]
        if version_info.get("is_current"):
            _supersede_older_versions(
                db, client_id, doc_id, version_info["version_group"]
            )

    # 7. Run Ingestion Pipeline
    # We inject business metadata into documents before running the pipeline.
    # Build content-signal lookup from parsed units (by index, since docs/units
    # were created in the same order inside parse_document()).
    for i, doc in enumerate(llama_docs):
        unit = units[i] if i < len(units) else {}
        doc.metadata.update(
            {
                **document_metadata,
                # Version awareness
                "version_label": version_info.get("version_label"),
                "document_version_group": version_info.get("version_group"),
                "effective_from": (
                    version_info.get("effective_from").isoformat()
                    if version_info.get("effective_from")
                    else None
                ),
                "effective_to": (
                    version_info.get("effective_to").isoformat()
                    if version_info.get("effective_to")
                    else None
                ),
                # Content signals (from parser heuristics)
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
                # Authority & provenance
                "authority_score": 1.0,
            }
        )

    layout_aware_pdf = (
        file_ext == ".pdf"
        and bool(llama_docs)
        and any((doc.metadata or {}).get("chunk_type") for doc in llama_docs)
    )
    transformations: list[Any] = []
    if not layout_aware_pdf:
        transformations.append(_build_non_layout_node_parser())
        logger.info("Using semantic splitter for %s", filename)
        try:
            transformations.extend(
                [
                    TitleExtractor(nodes=5),
                    SummaryExtractor(summaries=["prev", "self"]),
                    KeywordExtractor(keywords=10),
                    QuestionsAnsweredExtractor(num_questions=3),
                    DocumentContextExtractor(llm=LlamaSettings.llm, num_workers=3),
                ]
            )
            logger.info("Added LLM-based extractors (Title, Summary) to pipeline")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM extractors: {e}. Skipping.")
    else:
        logger.info(
            "Layout-aware PDF chunks detected for %s; skipping sentence splitting",
            filename,
        )

    pipeline = IngestionPipeline(transformations=transformations)

    # Run pipeline (includes chunking, metadata extraction, and vector indexing)
    worker_count = 1 if layout_aware_pdf else 3
    nodes: List[BaseNode] = pipeline.run(documents=llama_docs, num_workers=worker_count)

    _apply_retrieval_metadata(nodes, filename=filename, version_info=version_info)
    _apply_ref_doc_ids(nodes)

    # Exclude non-semantic metadata from embedding/LLM contexts while keeping
    # payload metadata available for filtering and citations.
    _apply_metadata_exclusions(nodes)

    # Persist nodes into Qdrant explicitly.
    vector_store_manager.index_nodes(nodes)

    qdrant_points = vector_store_manager.get_qdrant_point_count()
    if qdrant_points is not None:
        logger.info(
            "Qdrant collection '%s' current point count: %d",
            COLLECTION_NAME,
            qdrant_points,
        )

    # 8. Persist only vector-node registry mappings to SQL DB.
    for node in nodes:
        registry = VectorNodeRegistry(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            client_id=client_id,
            vector_collection=COLLECTION_NAME,
            vector_node_id=node.node_id,
            embedding_model=settings.EMBEDDING_MODEL,
        )
        db.add(registry)

    # 9. Mark success
    db_doc.status = "indexed"
    job.status = "completed"
    job.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(db_doc)

    logger.info(
        "Ingestion complete: doc=%s, parsed_units=%d, vector_registry_rows=%d",
        doc_id,
        len(units),
        len(nodes),
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
            DocumentVersion.is_current == True,
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
