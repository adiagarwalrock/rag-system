import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode
from sqlalchemy import true
from sqlalchemy.orm import Session

from llama_index.core import Settings as LlamaSettings
from app.core.config import settings
from app.db.snowflake import SessionLocal
from app.indexing.vector_store import COLLECTION_NAME, vector_store_manager
from llama_index.core.extractors import (
    DocumentContextExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)

from app.ingestion.parser import parse_document, save_upload_file
from app.db.models.client import Client
from app.db.models.document import (
    Document,
    DocumentVersion,
    IngestionJob,
    VectorNodeRegistry,
)
from app.ingestion.validator import (
    compute_checksum,
    validate_file_size,
    validate_file_type,
)
from app.ingestion.version_resolver import resolve_version

# --- Delegated Imports ---
from app.services.ingest_queue import IngestionQueueTask, get_ingestion_queue_manager
from app.services.ingest_metadata import (
    _apply_metadata_exclusions,
    _apply_ref_doc_ids,
    _apply_retrieval_metadata,
    _build_non_layout_node_parser,
)

logger = logging.getLogger(__name__)

RAW_DATA_DIR = settings.RAW_DATA_DIR


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
) -> IngestionJob:
    return IngestionJob(
        id=job_id,
        client_id=client_id,
        document_id=doc_id,
        status=status,
        started_at=started_at,
        parser_name="rag_ingestion_pipeline",
        parser_version="2.0.0",
        filesize_bytes=file_size,
    )


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

    if not db_doc.storage_path or not Path(db_doc.storage_path).exists():
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
    job = _create_ingestion_job_record(
        job_id=job_id,
        client_id=client_id,
        doc_id=doc_id,
        status="running",
        file_size=Path(file_path).stat().st_size,
        started_at=datetime.now(timezone.utc),
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
        if db_doc.storage_path and Path(db_doc.storage_path).exists():
            try:
                Path(db_doc.storage_path).unlink()
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


def _isoformat_or_none(value: Any) -> str | None:
    return value.isoformat() if value else None


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
        llama_docs, units = parse_document(self.file_path, document_metadata)
        self._apply_parser_metadata(llama_docs)

        version_info = self._resolve_version_info(llama_docs)
        self._persist_version_record(version_info)
        self._apply_document_metadata(
            llama_docs=llama_docs,
            units=units,
            document_metadata=document_metadata,
            version_info=version_info,
        )

        layout_aware_pdf = self._is_layout_aware_pdf(llama_docs)
        nodes = self._run_ingestion_pipeline(
            llama_docs=llama_docs,
            layout_aware_pdf=layout_aware_pdf,
        )

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

    def _apply_parser_metadata(self, llama_docs: list[Any]) -> None:
        if not llama_docs:
            return
        parser_name = llama_docs[0].metadata.get("parser_name")
        parser_version = llama_docs[0].metadata.get("parser_version")
        if parser_name:
            self.job.parser_name = parser_name
        if parser_version:
            self.job.parser_version = parser_version

    def _resolve_version_info(self, llama_docs: list[Any]) -> dict[str, Any]:
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
        llama_docs: list[Any],
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
                }
            )

    def _is_layout_aware_pdf(self, llama_docs: list[Any]) -> bool:
        return (
            self.file_ext == ".pdf"
            and bool(llama_docs)
            and any((doc.metadata or {}).get("chunk_type") for doc in llama_docs)
        )

    def _run_ingestion_pipeline(
        self,
        *,
        llama_docs: list[Any],
        layout_aware_pdf: bool,
    ) -> list[BaseNode]:
        transformations = self._build_transformations(layout_aware_pdf)
        pipeline = IngestionPipeline(transformations=transformations)
        worker_count = 1 if layout_aware_pdf else 3
        return pipeline.run(documents=llama_docs, num_workers=worker_count)

    def _build_transformations(self, layout_aware_pdf: bool) -> list[Any]:
        if layout_aware_pdf:
            logger.info(
                "Layout-aware PDF chunks detected for %s; skipping sentence splitting",
                self.filename,
            )
            return []

        transformations: list[Any] = [_build_non_layout_node_parser()]
        logger.info("Using semantic splitter for %s", self.filename)
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
        except Exception as exc:
            logger.warning("Failed to initialize LLM extractors: %s. Skipping.", exc)
        return transformations

    def _index_nodes(self, nodes: list[BaseNode]) -> None:
        vector_store_manager.index_nodes(nodes)
        qdrant_points = vector_store_manager.get_qdrant_point_count()
        if qdrant_points is None:
            return
        logger.info(
            "Qdrant collection '%s' current point count: %d",
            COLLECTION_NAME,
            qdrant_points,
        )

    def _persist_registry_rows(self, nodes: list[BaseNode]) -> None:
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
