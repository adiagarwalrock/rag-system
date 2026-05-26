from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.snowflake import get_db
from app.services.query_history_service import QueryHistoryFilters, QueryHistoryService

router = APIRouter()


@router.get("/{client_id}/history")
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


@router.get("/{client_id}/query-history")
def list_query_history(
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
