from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.snowflake import get_db
from app.services.query_history_service import QueryHistoryFilters, QueryHistoryService

router = APIRouter()


@router.get("/{client_id}/history")
@router.get("/{client_id}/query-history")
def list_history(
    client_id: str,
    status: str | None = Query(None),
    search_text: str | None = Query(None),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    return QueryHistoryService(db).list_query_history(
        QueryHistoryFilters(
            client_id=client_id,
            status=status,
            search_text=search_text,
            limit=limit,
            offset=offset,
        )
    )


@router.delete("/{client_id}/history/{query_id}")
@router.delete("/{client_id}/query-history/{query_id}")
def delete_history_item(
    client_id: str,
    query_id: str,
    db: Session = Depends(get_db),
):
    try:
        QueryHistoryService(db).delete_query_history_item(
            client_id=client_id,
            query_id=query_id,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Query history item not found")
    return {"status": "success", "query_id": query_id}
