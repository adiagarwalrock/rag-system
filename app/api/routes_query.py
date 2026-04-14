"""
Query API route — separate from documents.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_active_user
from app.db.models.client import Client
from app.db.models.user import User
from app.db.snowflake import get_db
from app.schemas.document import QueryRequest, QueryResponse
from app.services.query_service import execute_query

router = APIRouter()


@router.post("/", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Query documents scoped to a client."""
    # Validate client exists
    client = db.query(Client).filter(Client.id == request.client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    result = execute_query(
        question=request.question,
        client_id=request.client_id,
        user_id=current_user.id,
        db=db,
    )
    return QueryResponse(
        answer=result["answer"],
        reasoning=result.get("reasoning"),
        citations=result.get("citations", []),
        conflicts=result.get("conflicts", []),
        query_id=result.get("query_id"),
        latency_ms=result.get("latency_ms"),
        source_count=result.get("source_count", 0),
        evidence_count=result.get("evidence_count", 0),
        images_used=result.get("images_used", []),
        image_evidence_count=result.get("image_evidence_count", 0),
    )
