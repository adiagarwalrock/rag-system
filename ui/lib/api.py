"""
Core Python service interface for the RAG-System Streamlit UI.
This bypasses HTTP requests and talks directly natively to the app layers.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from app.db.models.client import Client
from app.db.models.document import (
    Document,
    DocumentVersion,
    IngestionJob,
    VectorNodeRegistry,
)
from app.db.snowflake import SessionLocal, engine
from app.db.schema import ensure_runtime_schema
from app.services.chat_conversation_service import ChatConversationService
from app.services.client_service import delete_client as delete_client_with_cascade
from app.services.ingest_service import (
    delete_document,
    enqueue_document_ingestion,
    retry_ingestion,
)
from app.services.query_history_service import QueryHistoryFilters, QueryHistoryService
from app.services.status_service import RuntimeStatusService

logger = logging.getLogger(__name__)


class VecteraCore:
    """
    Native internal wrapper mocking the old API schema for the frontend.
    """

    def __init__(self):
        # We don't maintain a single session here to avoid side effects across
        # separate streamlit interactions. We instantiate it per call.
        ensure_runtime_schema(engine)

    # --- Clients ---
    def list_clients(self) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            clients = db.query(Client).all()
            return [
                {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in clients
            ]

    def create_client(self, name: str, description: str = "") -> dict:
        with SessionLocal() as db:
            # check if exists
            existing = db.query(Client).filter(Client.name == name).first()
            if existing:
                raise ValueError(f"Client {name} already exists.")

            import uuid

            new_client = Client(
                id=str(uuid.uuid4()), name=name, description=description
            )
            db.add(new_client)
            db.commit()
            db.refresh(new_client)
            return {
                "id": new_client.id,
                "name": new_client.name,
                "description": new_client.description,
            }

    def delete_client(self, client_id: str) -> dict:
        with SessionLocal() as db:
            delete_client_with_cascade(client_id=client_id, db=db)
            return {
                "status": "success",
                "message": "Client deleted",
                "client_id": client_id,
            }

    # --- Documents ---
    def list_documents(self, client_id: str = None) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            query = db.query(Document)
            if client_id:
                query = query.filter(Document.client_id == client_id)
            documents = query.order_by(Document.created_at.desc()).all()
            results = []
            for d in documents:
                results.append(
                    {
                        "id": d.id,
                        "client_id": d.client_id,
                        "name": d.name,
                        "file_type": d.file_type,
                        "status": d.status,
                        "document_family": d.document_family,
                        "created_at": (
                            d.created_at.isoformat() if d.created_at else None
                        ),
                    }
                )
            return results

    def upload_document(
        self,
        client_id: str,
        file_name: str,
        file_content: bytes,
    ) -> dict:
        with SessionLocal() as db:
            # Need client_name for ingestion service metadata
            client = db.query(Client).filter(Client.id == client_id).first()
            client_name = client.name if client else "Unknown"

            doc, job = enqueue_document_ingestion(
                file_content=file_content,
                filename=file_name,
                client_id=client_id,
                client_name=client_name,
                db=db,
            )
            return {
                "id": doc.id,
                "name": doc.name,
                "status": doc.status,
                "ingestion_job_id": job.id,
            }

    def get_document_status(self, document_id: str) -> dict:
        with SessionLocal() as db:
            vector_row_count = (
                db.query(VectorNodeRegistry)
                .filter(
                    VectorNodeRegistry.document_id == document_id,
                    VectorNodeRegistry.is_active == True,
                )
                .count()
            )

            version = (
                db.query(DocumentVersion)
                .filter(DocumentVersion.document_id == document_id)
                .first()
            )

            job = (
                db.query(IngestionJob)
                .filter(IngestionJob.document_id == document_id)
                .order_by(IngestionJob.started_at.desc())
                .first()
            )
            if not job:
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc:
                    return {
                        "id": "legacy",
                        "document_id": document_id,
                        "status": doc.status,
                        "vector_point_count": vector_row_count,
                        "version_label": version.version_label if version else None,
                        "version_group": version.version_group if version else None,
                        "is_current_version": version.is_current if version else None,
                    }
                raise ValueError("Document not found")
            return {
                "id": job.id,
                "document_id": job.document_id,
                "status": job.status,
                "vector_point_count": vector_row_count,
                "error_message": job.error_message,
                "version_label": version.version_label if version else None,
                "version_group": version.version_group if version else None,
                "is_current_version": version.is_current if version else None,
            }

    def retry_document_ingestion(self, document_id: str) -> dict:
        with SessionLocal() as db:
            doc = retry_ingestion(document_id=document_id, db=db)
            return {"id": doc.id, "name": doc.name, "status": doc.status}

    def delete_document(self, document_id: str, hard: bool = False) -> dict:
        with SessionLocal() as db:
            delete_document(document_id=document_id, db=db, hard=hard)
            return {"status": "success", "message": "Document deleted"}

    # --- Query ---
    def query(
        self,
        client_id: str,
        question: str,
        reasoning_effort: str = "medium",
        reasoning_summary: str | None = None,
        session_id: str | None = None,
        status_callback=None,
        reasoning_callback=None,
    ) -> dict:
        with SessionLocal() as db:
            try:
                response = ChatConversationService(db).execute_client_query(
                    client_id=client_id,
                    question=question,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                    session_id=session_id,
                    status_callback=status_callback,
                    reasoning_callback=reasoning_callback,
                )
                return response
            except Exception as e:
                logger.error(f"Query Error: {e}")
                raise ValueError(str(e))

    def list_chat_sessions(self, client_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            sessions = ChatConversationService(db).list_sessions(
                client_id=client_id,
                limit=limit,
            )
            return [
                {
                    "id": session.id,
                    "client_id": session.client_id,
                    "title": session.title,
                    "summary_text": session.summary_text,
                    "created_at": (
                        session.created_at.isoformat() if session.created_at else None
                    ),
                    "updated_at": (
                        session.updated_at.isoformat() if session.updated_at else None
                    ),
                    "last_activity_at": (
                        session.last_activity_at.isoformat()
                        if session.last_activity_at
                        else None
                    ),
                }
                for session in sessions
            ]

    def create_chat_session(self, client_id: str, title: str | None = None) -> Dict[str, Any]:
        with SessionLocal() as db:
            session = ChatConversationService(db).create_session(
                client_id=client_id, title=title
            )
            return {
                "id": session.id,
                "client_id": session.client_id,
                "title": session.title,
                "summary_text": session.summary_text,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                "last_activity_at": (
                    session.last_activity_at.isoformat()
                    if session.last_activity_at
                    else None
                ),
            }

    def list_chat_messages(self, session_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            messages = ChatConversationService(db).list_messages(
                session_id=session_id, limit=limit
            )
            response_messages: List[Dict[str, Any]] = []
            for message in messages:
                payload: Dict[str, Any] = {
                    "id": message.id,
                    "client_id": message.client_id,
                    "session_id": message.session_id,
                    "role": message.role,
                    "content": message.content,
                    "turn_index": message.turn_index,
                    "query_log_id": message.query_log_id,
                    "created_at": (
                        message.created_at.isoformat() if message.created_at else None
                    ),
                }
                if message.role == "assistant":
                    citations: List[Dict[str, Any]] = []
                    if message.citations_json:
                        try:
                            decoded = json.loads(message.citations_json)
                            if isinstance(decoded, list):
                                citations = [
                                    item
                                    for item in decoded
                                    if isinstance(item, dict)
                                ]
                        except Exception:
                            citations = []

                    reasoning = (message.reasoning or "").strip() or None
                    if reasoning or citations or message.query_log_id:
                        payload["result"] = {
                            "reasoning": reasoning,
                            "citations": citations,
                            "query_id": message.query_log_id,
                        }

                response_messages.append(payload)
            return response_messages

    def clear_chat_session(self, session_id: str) -> Dict[str, Any]:
        with SessionLocal() as db:
            ChatConversationService(db).clear_session(session_id=session_id)
            return {"status": "success", "session_id": session_id}

    def list_query_history(
        self,
        client_id: Optional[str] = None,
        status: Optional[str] = None,
        search_text: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        with SessionLocal() as db:
            payload = QueryHistoryService(db).list_query_history(
                QueryHistoryFilters(
                    client_id=client_id,
                    status=status,
                    search_text=search_text,
                    limit=limit,
                    offset=offset,
                )
            )
            rows = []
            for row in payload["rows"]:
                rows.append(
                    {
                        **row,
                        "created_at": (
                            row["created_at"].isoformat()
                            if row.get("created_at")
                            else None
                        ),
                    }
                )
            return {"rows": rows, "total": payload["total"]}

    # --- Health ---
    def get_runtime_status(self) -> dict:
        return RuntimeStatusService().get_status()

    def health(self) -> dict:
        try:
            with SessionLocal() as db:
                db.execute(__import__("sqlalchemy").text("SELECT 1"))
            return {"status": "ok"}
        except Exception:
            return {"status": "error"}
