import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_active_user
from app.db.models.client import Client
from app.db.models.user import User
from app.db.snowflake import get_db
from app.schemas.client import ClientCreate, ClientResponse, ClientUpdate

router = APIRouter()


@router.get("/", response_model=List[ClientResponse])
def get_clients(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)
):
    return db.query(Client).filter(Client.is_active == True).all()


@router.get("/{client_id}", response_model=ClientResponse)
def get_client(
    client_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    return client


@router.post("/", response_model=ClientResponse)
def create_client(
    client_in: ClientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    db_client = Client(
        id=str(uuid.uuid4()),
        name=client_in.name,
        description=client_in.description,
        created_by=current_user.id,
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
    current_user: User = Depends(get_current_active_user),
):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    update_data = updates.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(client, key, value)

    db.commit()
    db.refresh(client)
    return client
