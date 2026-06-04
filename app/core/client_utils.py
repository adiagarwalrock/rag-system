"""Shared utilities for per-client configuration resolution."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models.client import Client


def resolve_client_embedding_model(client_id: str, db: Session) -> str:
    """Return the embedding model configured for a client, or the global default.

    Returns settings.EMBEDDING_MODEL when the client has no model set or the
    client_id is not found — ensures backward compatibility for existing clients.
    """
    client = db.get(Client, client_id)
    return (client.embedding_model if client is not None else None) or settings.EMBEDDING_MODEL
