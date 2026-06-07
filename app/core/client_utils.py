"""Shared utilities for per-client configuration resolution."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models.client import Client


def _resolve_client_field(client_id: str, db: Session, attr: str, default: str) -> str:
    client = db.get(Client, client_id)
    return (getattr(client, attr) if client is not None else None) or default


def resolve_client_embedding_model(client_id: str, db: Session) -> str:
    """Return the embedding model configured for a client, or the global default."""
    return _resolve_client_field(client_id, db, "embedding_model", settings.EMBEDDING_MODEL)


def resolve_client_llm_model(client_id: str, db: Session) -> str:
    """Return the LLM model configured for a client, or the global default."""
    return _resolve_client_field(client_id, db, "llm_model", settings.LLM_MODEL)
