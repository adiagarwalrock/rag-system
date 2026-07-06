"""
Document and ingestion API routes.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import func, true
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models.client import Client
from app.db.models.document import (
    Document,
    DocumentVersion,
    IngestionJob,
    VectorNodeRegistry,
)
from app.db.snowflake import get_db
from app.ingestion.parser.registry import (
    get_available_parsers,
    validate_parser_preference,
)
from app.schemas.document import (
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatusResponse,
    ParserListResponse,
)
from app.services.client_service import ClientLookupService
from app.services.ingest_service import (
    delete_document,
    enqueue_document_ingestion,
    retry_ingestion,
)

router = APIRouter()
_LEGACY_JOB_PARSER_NAME = "rag_ingestion_pipeline"


@router.get("/", response_model=List[DocumentListResponse])
def list_documents(
    client_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """List documents, optionally filtered by client."""
    query = db.query(Document)
    if client_id:
        query = query.filter(Document.client_id == client_id)
    documents = query.order_by(Document.created_at.desc()).all()
    return _enrich_document_list(documents, db)


@router.get("/parsers", response_model=ParserListResponse)
def list_parsers():
    """List available document parsers given current server configuration."""
    return {"parsers": get_available_parsers()}


def _enrich_document_list(
    documents: list[Document],
    db: Session,
) -> list[dict]:
    if not documents:
        return []

    document_ids = [document.id for document in documents]
    vector_counts = {
        row.document_id: int(row.count or 0)
        for row in (
            db.query(
                VectorNodeRegistry.document_id,
                func.count(VectorNodeRegistry.id).label("count"),
            )
            .filter(
                VectorNodeRegistry.document_id.in_(document_ids),
                VectorNodeRegistry.is_active == true(),
            )
            .group_by(VectorNodeRegistry.document_id)
            .all()
        )
    }

    # Embedding model used to index each document (from active VectorNodeRegistry rows)
    doc_embedding_models: dict[str, str] = {}
    emb_rows = (
        db.query(VectorNodeRegistry.document_id, VectorNodeRegistry.embedding_model)
        .filter(
            VectorNodeRegistry.document_id.in_(document_ids),
            VectorNodeRegistry.is_active == true(),
            VectorNodeRegistry.embedding_model != None,  # noqa: E711
        )
        .distinct()
        .all()
    )
    for row in emb_rows:
        doc_embedding_models.setdefault(row.document_id, row.embedding_model)

    # Current embedding model per client for staleness check
    client_ids = {doc.client_id for doc in documents}
    clients = db.query(Client).filter(Client.id.in_(client_ids)).all()
    client_models: dict[str, str] = {
        c.id: (c.embedding_model or settings.EMBEDDING_MODEL) for c in clients
    }

    latest_parser_by_doc: dict[str, str | None] = {}
    jobs = (
        db.query(IngestionJob)
        .filter(IngestionJob.document_id.in_(document_ids))
        .order_by(IngestionJob.started_at.desc(), IngestionJob.id.desc())
        .all()
    )
    for job in jobs:
        if job.document_id in latest_parser_by_doc:
            continue
        latest_parser_by_doc[job.document_id] = _display_parser_name(job.parser_name)

    result = []
    for document in documents:
        doc_model = doc_embedding_models.get(document.id)
        client_model = client_models.get(document.client_id)
        result.append(
            {
                "id": document.id,
                "client_id": document.client_id,
                "name": document.name,
                "file_type": document.file_type,
                "status": document.status,
                "document_family": document.document_family,
                "parser_used": latest_parser_by_doc.get(document.id),
                "vector_point_count": vector_counts.get(document.id, 0),
                "embedding_model": doc_model,
                "embedding_model_stale": bool(
                    doc_model and client_model and doc_model != client_model
                ),
                "created_at": document.created_at,
            }
        )
    return result


def _display_parser_name(parser_name: str | None) -> str | None:
    if not parser_name or parser_name == _LEGACY_JOB_PARSER_NAME:
        return None
    return parser_name


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
):
    """Get a single document by ID."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(
    document_id: str,
    db: Session = Depends(get_db),
):
    """Get ingestion status for a document."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get version info
    version = (
        db.query(DocumentVersion)
        .filter(DocumentVersion.document_id == document_id)
        .first()
    )

    vector_point_count = (
        db.query(VectorNodeRegistry)
        .filter(
            VectorNodeRegistry.document_id == document_id,
            VectorNodeRegistry.is_active == true(),
        )
        .count()
    )

    latest_job = (
        db.query(IngestionJob)
        .filter(IngestionJob.document_id == document_id)
        .order_by(IngestionJob.started_at.desc(), IngestionJob.id.desc())
        .first()
    )

    return {
        "document_id": doc.id,
        "name": doc.name,
        "status": doc.status,
        "ingestion_job_id": latest_job.id if latest_job else None,
        "ingestion_job_status": latest_job.status if latest_job else None,
        "vector_point_count": vector_point_count,
        "document_family": doc.document_family,
        "version_label": version.version_label if version else None,
        "version_group": version.version_group if version else None,
        "is_current_version": version.is_current if version else None,
    }


@router.post("/ingest", response_model=DocumentResponse)
async def ingest_doc(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    parser: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """Upload and ingest a document."""
    try:
        validate_parser_preference(parser)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        client = ClientLookupService(db).require_client(client_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Client not found")

    # Read file content
    file_content = await file.read()

    result, job = enqueue_document_ingestion(
        file_content=file_content,
        filename=file.filename,
        client_id=client_id,
        client_name=client.name,
        db=db,
        parser_preference=parser,
    )
    return {
        "id": result.id,
        "client_id": result.client_id,
        "name": result.name,
        "file_type": result.file_type,
        "status": result.status,
        "checksum": result.checksum,
        "document_family": result.document_family,
        "created_at": result.created_at,
        "updated_at": result.updated_at,
        "ingestion_job_id": job.id,
    }


@router.post("/{document_id}/retry", response_model=DocumentResponse)
def retry_doc_ingestion(
    document_id: str,
    parser: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """Retry ingestion for a failed document. Pass ?parser=<id> to override the parser."""
    try:
        result = retry_ingestion(
            document_id=document_id, db=db, parser_preference=parser
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_doc(
    document_id: str,
    hard: bool = Query(False),
    db: Session = Depends(get_db),
):
    """Delete a document and its associated vectors/data."""
    try:
        delete_document(document_id=document_id, db=db, hard=hard)
        return {"status": "success", "message": "Document deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
