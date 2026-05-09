"""
Query API route — separate from documents.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.snowflake import get_db
from app.schemas.document import QueryRequest, QueryResponse
from app.services.chat_conversation_service import ChatConversationService
from app.services.client_service import ClientLookupService

router = APIRouter()


@router.post("/", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db),
):
    """Query documents scoped to a client."""
    try:
        ClientLookupService(db).require_client(request.client_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Client not found")

    result = ChatConversationService(db).execute_client_query(
        question=request.question,
        client_id=request.client_id,
        reasoning_effort=request.reasoning_effort,
        session_id=request.session_id,
    )
    return QueryResponse(
        answer=result["answer"],
        reasoning=str(result.get("reasoning") or ""),
        citations=result.get("citations", []),
        conflicts=result.get("conflicts", []),
        query_id=result.get("query_id"),
        latency_ms=result.get("latency_ms"),
        source_count=result.get("source_count", 0),
        evidence_count=result.get("evidence_count", 0),
        images_used=result.get("images_used", []),
        image_evidence_count=result.get("image_evidence_count", 0),
        reasoning_effort=result.get("reasoning_effort", "medium"),
        reasoning_effort_applied=result.get("reasoning_effort_applied", False),
        session_id=result.get("session_id"),
        user_message_id=result.get("user_message_id"),
        assistant_message_id=result.get("assistant_message_id"),
    )
