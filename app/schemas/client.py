from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ClientBase(BaseModel):
    name: str
    description: Optional[str] = None
    is_active: bool = True
    embedding_model: Optional[str] = None


class ClientCreate(ClientBase):
    pass


class ClientUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    embedding_model: Optional[str] = None


class ClientResponse(ClientBase):
    id: str
    created_at: datetime
    updated_at: datetime
    document_count: Optional[int] = None
    query_count: Optional[int] = None
    session_count: Optional[int] = None
    memory_point_count: Optional[int] = None

    class Config:
        from_attributes = True
