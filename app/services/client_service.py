"""
Client service: workflows related to client lifecycle operations.
"""


import logging

from sqlalchemy.orm import Session

from app.db.models.chat import ChatMessage, ChatSession
from app.db.models.client import Client
from app.db.models.document import (
    ConflictLog,
    Document,
    DocumentVersion,
    IngestionJob,
    QueryLog,
    RetrievalLog,
    VectorNodeRegistry,
)
from app.indexing.chat_history_store import chat_history_store
from app.services.ingest_service import delete_document

logger = logging.getLogger(__name__)


class ClientDeletionService:
    """Delete a client and all associated records across SQL and vector store."""

    def __init__(self, db: Session):
        self.db = db

    def delete_client(self, client_id: str) -> None:
        client = self._get_client(client_id)
        document_ids = self._list_client_document_ids(client_id)
        logger.info(
            "Deleting client '%s' with %d documents.",
            client_id,
            len(document_ids),
        )

        try:
            self._delete_client_documents(document_ids)
            self._delete_query_history(client_id)
            self._delete_client_residual_rows(client_id, document_ids)
            self.db.delete(client)
            self.db.commit()
            logger.info("Deleted client '%s' and associated records.", client_id)
        except Exception as exc:
            self.db.rollback()
            raise ValueError(f"Failed to delete client {client_id}: {exc}") from exc

    def _get_client(self, client_id: str) -> Client:
        client = self.db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise ValueError(f"Client {client_id} not found.")
        return client

    def _list_client_document_ids(self, client_id: str) -> list[str]:
        rows = self.db.query(Document.id).filter(Document.client_id == client_id).all()
        return [row[0] for row in rows]

    def _delete_client_documents(self, document_ids: list[str]) -> None:
        for document_id in document_ids:
            # Reuse the existing document hard-delete path to keep SQL + vector cleanup
            # behavior consistent in one place.
            delete_document(document_id=document_id, db=self.db, hard=True)

    def _delete_query_history(self, client_id: str) -> None:
        query_log_ids = [
            row[0]
            for row in self.db.query(QueryLog.id)
            .filter(QueryLog.client_id == client_id)
            .all()
        ]

        if query_log_ids:
            self.db.query(RetrievalLog).filter(
                RetrievalLog.query_log_id.in_(query_log_ids)
            ).delete(synchronize_session=False)
            self.db.query(ConflictLog).filter(
                ConflictLog.query_log_id.in_(query_log_ids)
            ).delete(synchronize_session=False)

        self.db.query(ChatMessage).filter(ChatMessage.client_id == client_id).delete(
            synchronize_session=False
        )
        self.db.query(QueryLog).filter(QueryLog.client_id == client_id).delete(
            synchronize_session=False
        )
        self.db.query(ChatSession).filter(ChatSession.client_id == client_id).delete(
            synchronize_session=False
        )
        chat_history_store.delete_client(client_id)

    def _delete_client_residual_rows(
        self, client_id: str, document_ids: list[str]
    ) -> None:
        if document_ids:
            self.db.query(DocumentVersion).filter(
                DocumentVersion.document_id.in_(document_ids)
            ).delete(synchronize_session=False)

        self.db.query(VectorNodeRegistry).filter(
            VectorNodeRegistry.client_id == client_id
        ).delete(synchronize_session=False)
        self.db.query(IngestionJob).filter(IngestionJob.client_id == client_id).delete(
            synchronize_session=False
        )
        self.db.query(Document).filter(Document.client_id == client_id).delete(
            synchronize_session=False
        )


def delete_client(client_id: str, db: Session) -> None:
    ClientDeletionService(db).delete_client(client_id)


class ClientLookupService:
    """Read-only client lookups shared by API and service workflows."""

    def __init__(self, db: Session):
        self.db = db

    def get_client(self, client_id: str) -> Client | None:
        return self.db.query(Client).filter(Client.id == client_id).first()

    def get_client_by_name(self, client_name: str) -> Client | None:
        cleaned_name = client_name.strip()
        if not cleaned_name:
            return None
        return self.db.query(Client).filter(Client.name == cleaned_name).first()

    def require_client(self, client_id: str) -> Client:
        client = self.get_client(client_id)
        if not client:
            raise ValueError(f"Client {client_id} not found")
        return client
