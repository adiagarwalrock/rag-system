import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import true
from sqlalchemy.orm import Session

from app.db.models.client import Client
from app.db.snowflake import get_db
from app.schemas.client import ClientCreate, ClientResponse, ClientUpdate
from app.services.client_service import (
    ClientLookupService,
    delete_client as delete_client_with_cascade,
)

router = APIRouter()


@router.get("/", response_model=List[ClientResponse])
def get_clients(db: Session = Depends(get_db)):
    return db.query(Client).filter(Client.is_active == true()).all()


@router.get("/{client_id}", response_model=ClientResponse)
def get_client(
    client_id: str,
    db: Session = Depends(get_db),
):
    try:
        client = ClientLookupService(db).require_client(client_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Client not found")
    return client


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
