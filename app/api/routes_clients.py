import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, true
from sqlalchemy.orm import Session

from app.db.models.chat import ChatSession
from app.db.models.client import Client
from app.db.models.document import Document, QueryLog, VectorNodeRegistry
from app.db.snowflake import get_db
from app.schemas.client import ClientCreate, ClientResponse, ClientUpdate
from app.services.client_service import (
    ClientLookupService,
    delete_client as delete_client_with_cascade,
)

router = APIRouter()
ZERO_CLIENT_COUNTS = {
    "document_count": 0,
    "query_count": 0,
    "session_count": 0,
    "memory_point_count": 0,
}


def _count_by_client(db: Session, model: Any, client_ids: list[str], *extra_filters: Any) -> dict[str, int]:
    q = (
        db.query(model.client_id, func.count(model.id).label("cnt"))
        .filter(model.client_id.in_(client_ids), *extra_filters)
        .group_by(model.client_id)
    )
    return {row.client_id: row.cnt for row in q.all()}


def _client_counts(db: Session, client_ids: list[str]) -> dict[str, dict]:
    """Return per-client counts for documents, queries, sessions, and vector nodes."""
    if not client_ids:
        return {}

    doc_counts = _count_by_client(db, Document, client_ids)
    query_counts = _count_by_client(db, QueryLog, client_ids)
    session_counts = _count_by_client(db, ChatSession, client_ids)
    memory_counts = _count_by_client(
        db, VectorNodeRegistry, client_ids, VectorNodeRegistry.is_active == true()
    )

    return {
        cid: {
            "document_count": doc_counts.get(cid, 0),
            "query_count": query_counts.get(cid, 0),
            "session_count": session_counts.get(cid, 0),
            "memory_point_count": memory_counts.get(cid, 0),
        }
        for cid in client_ids
    }


def _enrich(client: Client, counts: dict) -> ClientResponse:
    data = {
        "id": client.id,
        "name": client.name,
        "description": client.description,
        "is_active": client.is_active,
        "created_at": client.created_at,
        "updated_at": client.updated_at,
        **ZERO_CLIENT_COUNTS,
        **counts,
    }
    return ClientResponse(**data)


@router.get("/", response_model=List[ClientResponse])
def get_clients(db: Session = Depends(get_db)):
    clients = db.query(Client).filter(Client.is_active == true()).all()
    counts = _client_counts(db, [c.id for c in clients])
    return [_enrich(c, counts.get(c.id, {})) for c in clients]


@router.get("/{client_id}", response_model=ClientResponse)
def get_client(
    client_id: str,
    db: Session = Depends(get_db),
):
    try:
        client = ClientLookupService(db).require_client(client_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Client not found")
    counts = _client_counts(db, [client_id])
    return _enrich(client, counts.get(client_id, {}))


@router.post("/", response_model=ClientResponse)
def create_client(
    client_in: ClientCreate,
    db: Session = Depends(get_db),
):
    db_client = Client(
        id=str(uuid.uuid4()),
        name=client_in.name,
        description=client_in.description,
    )
    db.add(db_client)
    db.commit()
    db.refresh(db_client)
    return _enrich(db_client, ZERO_CLIENT_COUNTS)


@router.patch("/{client_id}", response_model=ClientResponse)
def update_client(
    client_id: str,
    updates: ClientUpdate,
    db: Session = Depends(get_db),
):
    try:
        client = ClientLookupService(db).require_client(client_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Client not found")

    update_data = updates.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(client, key, value)

    db.commit()
    db.refresh(client)
    counts = _client_counts(db, [client_id])
    return _enrich(client, counts.get(client_id, {}))


@router.delete("/{client_id}")
def delete_client(
    client_id: str,
    db: Session = Depends(get_db),
):
    try:
        delete_client_with_cascade(client_id=client_id, db=db)
        return {
            "status": "success",
            "message": "Client deleted",
            "client_id": client_id,
        }
    except ValueError as e:
        detail = str(e)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail)
