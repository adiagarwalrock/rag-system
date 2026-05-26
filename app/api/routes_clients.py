import uuid
from typing import List

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


def _client_counts(db: Session, client_ids: list[str]) -> dict[str, dict]:
    """Return per-client counts for documents, queries, sessions, and vector nodes."""
    if not client_ids:
        return {}

    doc_counts = {
        row.client_id: row.cnt
        for row in db.query(Document.client_id, func.count(Document.id).label("cnt"))
        .filter(Document.client_id.in_(client_ids))
        .group_by(Document.client_id)
        .all()
    }
    query_counts = {
        row.client_id: row.cnt
        for row in db.query(QueryLog.client_id, func.count(QueryLog.id).label("cnt"))
        .filter(QueryLog.client_id.in_(client_ids))
        .group_by(QueryLog.client_id)
        .all()
    }
    session_counts = {
        row.client_id: row.cnt
        for row in db.query(
            ChatSession.client_id, func.count(ChatSession.id).label("cnt")
        )
        .filter(ChatSession.client_id.in_(client_ids))
        .group_by(ChatSession.client_id)
        .all()
    }
    memory_counts = {
        row.client_id: row.cnt
        for row in db.query(
            VectorNodeRegistry.client_id,
            func.count(VectorNodeRegistry.id).label("cnt"),
        )
        .filter(
            VectorNodeRegistry.client_id.in_(client_ids),
            VectorNodeRegistry.is_active == true(),
        )
        .group_by(VectorNodeRegistry.client_id)
        .all()
    }

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
    return db_client


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
    return client


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
