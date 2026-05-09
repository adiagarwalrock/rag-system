"""
Document and ingestion API routes.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import true
from sqlalchemy.orm import Session

from app.db.models.document import (
    Document,
    DocumentVersion,
    IngestionJob,
    VectorNodeRegistry,
)
from app.db.snowflake import get_db
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.client_service import ClientLookupService
from app.services.ingest_service import (
    delete_document,
    enqueue_document_ingestion,
    retry_ingestion,
)

router = APIRouter()


@router.get("/", response_model=List[DocumentListResponse])
def list_documents(
    client_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """List documents, optionally filtered by client."""
    query = db.query(Document)
    if client_id:
        query = query.filter(Document.client_id == client_id)
    return query.order_by(Document.created_at.desc()).all()


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


@router.get("/{document_id}/status")
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
    db: Session = Depends(get_db),
):
    """Upload and ingest a document."""
    try:
        client = ClientLookupService(db).require_client(client_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Client not found")

    # Read file content
    file_content = await file.read()

    result, job = enqueue_document_ingestion(
        file_content=file_content,
        filename=file.filename or "",
        client_id=client_id,
        client_name=str(client.name),
        db=db,
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
    db: Session = Depends(get_db),
):
    """Retry ingestion for a failed document."""
    try:
        result = retry_ingestion(document_id=document_id, db=db)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{document_id}")
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
