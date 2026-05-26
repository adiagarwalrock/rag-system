from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models.chat import ChatMessage, ChatSession
from app.db.snowflake import get_db
from app.services.chat_conversation_service import ChatConversationService
from app.services.client_service import ClientLookupService

router = APIRouter()


class ChatSessionCreate(BaseModel):
    title: str | None = None


def _serialize_session(session: ChatSession) -> dict[str, Any]:
    return {
        "id": session.id,
        "client_id": session.client_id,
        "title": session.title,
        "summary_text": session.summary_text,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        "last_activity_at": (
            session.last_activity_at.isoformat() if session.last_activity_at else None
        ),
    }


def _serialize_message(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": message.id,
        "client_id": message.client_id,
        "session_id": message.session_id,
        "role": message.role,
        "content": message.content,
        "turn_index": message.turn_index,
        "query_log_id": message.query_log_id,
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }

    if message.role == "assistant":
        citations: list[dict[str, Any]] = []
        if message.citations_json:
            try:
                decoded = json.loads(message.citations_json)
                if isinstance(decoded, list):
                    citations = [item for item in decoded if isinstance(item, dict)]
            except Exception:
                citations = []

        payload["result"] = {
            "reasoning": (message.reasoning or "").strip() or None,
            "citations": citations,
            "query_id": message.query_log_id,
        }

    return payload


@router.get("/{client_id}/sessions")
def list_sessions(
    client_id: str,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    try:
        ClientLookupService(db).require_client(client_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Client not found")

    sessions = ChatConversationService(db).list_sessions(
        client_id=client_id,
        limit=limit,
    )
    return [_serialize_session(session) for session in sessions]


@router.post("/{client_id}/sessions")
def create_session(
    client_id: str,
    payload: ChatSessionCreate | None = None,
    db: Session = Depends(get_db),
):
    try:
        ClientLookupService(db).require_client(client_id)
        session = ChatConversationService(db).create_session(
            client_id=client_id,
            title=payload.title if payload else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _serialize_session(session)


@router.delete("/{client_id}/sessions/{session_id}")
def delete_session(
    client_id: str,
    session_id: str,
    db: Session = Depends(get_db),
):
    try:
        ChatConversationService(db).delete_session(
            session_id=session_id, client_id=client_id
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "session_id": session_id}


@router.get("/{client_id}/sessions/{session_id}/messages")
def list_messages(
    client_id: str,
    session_id: str,
    limit: int = Query(200, ge=1, le=500),
    db: Session = Depends(get_db),
):
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.client_id == client_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = ChatConversationService(db).list_messages(
        session_id=session_id,
        limit=limit,
    )
    return [_serialize_message(message) for message in messages]


@router.delete("/{client_id}/sessions/{session_id}/messages")
def clear_messages(
    client_id: str,
    session_id: str,
    db: Session = Depends(get_db),
):
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.client_id == client_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    ChatConversationService(db).clear_session(session_id=session_id)
    return {"status": "success", "session_id": session_id}
