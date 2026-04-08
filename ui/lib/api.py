"""
Core Python service interface for the RAG-System Streamlit UI.
This bypasses HTTP requests and talks directly natively to the app layers.
"""

import logging
from typing import Any, Dict, List, Optional

from app.core.security import verify_password
from app.db.models.client import Client
from app.db.models.document import (
    Document,
    DocumentVersion,
    IngestionJob,
    VectorNodeRegistry,
)
from app.db.models.user import User
from app.db.snowflake import SessionLocal
from app.services.ingest_service import (
    ingest_document,
    retry_ingestion,
    delete_document,
)
from app.services.query_service import execute_query

logger = logging.getLogger(__name__)


class VecteraCore:
    """
    Native internal wrapper mocking the old API schema for the frontend.
    """

    def __init__(self):
        # We don't maintain a single session here to avoid side effects across
        # separate streamlit interactions. We instantiate it per call.
        pass

    # --- Auth ---
    def login(self, email: str, password: str) -> dict:
        """Login natively by comparing password hash."""
        with SessionLocal() as db:
            user = db.query(User).filter(User.email == email).first()
            if not user or not verify_password(password, user.password_hash):
                raise ValueError("Incorrect email or password")
            # Create a mock token data so the UI thinks auth succeeded
            return {"access_token": user.id, "token_type": "internal"}

    def get_me(self, user_id: str) -> dict:
        """Fetch user by id (derived from mock internal token)"""
        if not user_id:
            raise ValueError("Not authenticated")
        with SessionLocal() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found")
            return {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "role": "admin",  # Hardcoded default role since it depends on clients
            }

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

    # --- Documents ---
    def list_documents(self, client_id: str = None) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            query = db.query(Document)
            if client_id:
                query = query.filter(Document.client_id == client_id)
            documents = query.all()
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
        user_id: str = "internal",
    ) -> dict:
        with SessionLocal() as db:
            # Need client_name for ingestion service metadata
            client = db.query(Client).filter(Client.id == client_id).first()
            client_name = client.name if client else "Unknown"

            doc = ingest_document(
                file_content=file_content,
                filename=file_name,
                client_id=client_id,
                client_name=client_name,
                user_id=user_id,
                db=db,
            )
            return {"id": doc.id, "name": doc.name, "status": doc.status}

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

    def retry_document_ingestion(
        self, document_id: str, user_id: str = "internal"
    ) -> dict:
        with SessionLocal() as db:
            doc = retry_ingestion(document_id=document_id, user_id=user_id, db=db)
            return {"id": doc.id, "name": doc.name, "status": doc.status}

    def delete_document(
        self, document_id: str, hard: bool = False, user_id: str = "internal"
    ) -> dict:
        with SessionLocal() as db:
            delete_document(document_id=document_id, db=db, hard=hard)
            return {"status": "success", "message": "Document deleted"}

    # --- Query ---
    def query(self, client_id: str, question: str, user_id: str = "internal") -> dict:
        with SessionLocal() as db:
            try:
                response = execute_query(
                    client_id=client_id, question=question, user_id=user_id, db=db
                )
                return response
            except Exception as e:
                logger.error(f"Query Error: {e}")
                raise ValueError(str(e))

    # --- Health ---
    def health(self) -> dict:
        try:
            with SessionLocal() as db:
                db.execute(__import__("sqlalchemy").text("SELECT 1"))
            return {"status": "ok"}
        except Exception:
            return {"status": "error"}
