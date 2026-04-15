"""
Document and ingestion API routes.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_active_user
from app.db.models.client import Client
from app.db.models.document import Document, DocumentVersion, VectorNodeRegistry
from app.db.models.user import User
from app.db.snowflake import get_db
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.ingest_service import (
    delete_document,
    ingest_document,
    retry_ingestion,
)

router = APIRouter()


@router.get("/", response_model=List[DocumentListResponse])
def list_documents(
    client_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
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
    current_user: User = Depends(get_current_active_user),
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
    current_user: User = Depends(get_current_active_user),
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
            VectorNodeRegistry.is_active == True,
        )
        .count()
    )

    return {
        "document_id": doc.id,
        "name": doc.name,
        "status": doc.status,
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
    current_user: User = Depends(get_current_active_user),
):
    """Upload and ingest a document."""
    # Validate client exists
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Read file content
    file_content = await file.read()

    result = ingest_document(
        file_content=file_content,
        filename=file.filename,
        client_id=client_id,
        client_name=client.name,
        user_id=current_user.id,
        db=db,
    )
    return result


@router.post("/{document_id}/retry", response_model=DocumentResponse)
def retry_doc_ingestion(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retry ingestion for a failed document."""
    try:
        result = retry_ingestion(
            document_id=document_id, user_id=current_user.id, db=db
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{document_id}")
def delete_doc(
    document_id: str,
    hard: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a document and its associated vectors/data."""
    try:
        delete_document(document_id=document_id, db=db, hard=hard)
        return {"status": "success", "message": "Document deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
