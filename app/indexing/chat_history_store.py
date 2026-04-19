from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from qdrant_client.http import models as qdrant_models

from app.core.ai_provider import get_embeddings
from app.core.config import settings
from app.indexing.vector_store import vector_store_manager

logger = logging.getLogger(__name__)

CHAT_HISTORY_COLLECTION_NAME = settings.CHAT_HISTORY_COLLECTION_NAME
VECTOR_DIMENSIONS = settings.effective_vector_dimensions


@dataclass(frozen=True)
class ChatHistoryMatch:
    score: float
    client_id: str
    session_id: str
    user_text: str
    assistant_text: str
    assistant_message_id: str | None
    created_at: str | None


class ChatHistoryVectorStore:
    """Client-scoped semantic memory backed by Qdrant."""

    def __init__(self) -> None:
        self._embed_model = None
        self._collection_checked = False

    def index_qa_pair(
        self,
        *,
        point_id: str,
        client_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str,
        assistant_message_id: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        if not user_text.strip() or not assistant_text.strip():
            return

        try:
            self._ensure_collection()
            vector = self._embed_text(
                f"User:\n{user_text.strip()}\n\nAssistant:\n{assistant_text.strip()}"
            )
            if not vector:
                return

            payload = {
                "client_id": client_id,
                "session_id": session_id,
                "user_text": user_text,
                "assistant_text": assistant_text,
                "assistant_message_id": assistant_message_id,
                "created_at": ((created_at or datetime.now(timezone.utc)).isoformat()),
                "role_pair": "user_assistant",
            }
            point = qdrant_models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            vector_store_manager.get_qdrant_client().upsert(
                collection_name=CHAT_HISTORY_COLLECTION_NAME,
                points=[point],
                wait=False,
            )
        except Exception:
            logger.exception(
                "Failed to index chat history point for client %s", client_id
            )

    def search(
        self,
        *,
        client_id: str,
        query_text: str,
        limit: int,
        exclude_session_id: str | None = None,
    ) -> list[ChatHistoryMatch]:
        if not query_text.strip() or limit <= 0:
            return []

        try:
            self._ensure_collection()
            vector = self._embed_text(query_text)
            if not vector:
                return []

            points = self._search_points(
                client_id=client_id,
                vector=vector,
                limit=limit,
                exclude_session_id=exclude_session_id,
            )

            matches: list[ChatHistoryMatch] = []
            for point in points:
                match = self._point_to_match(point)
                if match is not None:
                    matches.append(match)
            return matches
        except Exception:
            logger.exception(
                "Failed semantic chat history search for client %s", client_id
            )
            return []

    def delete_client(self, client_id: str) -> None:
        try:
            self._ensure_collection()
            self._delete_by_field(field_name="client_id", value=client_id)
        except Exception:
            logger.exception(
                "Failed deleting chat history vectors for client %s", client_id
            )

    def delete_session(self, session_id: str) -> None:
        try:
            self._ensure_collection()
            self._delete_by_field(field_name="session_id", value=session_id)
        except Exception:
            logger.exception(
                "Failed deleting chat history vectors for session %s", session_id
            )

    def _search_points(
        self,
        *,
        client_id: str,
        vector: list[float],
        limit: int,
        exclude_session_id: str | None,
    ) -> list[Any]:
        query_filter = self._build_search_filter(
            client_id=client_id,
            exclude_session_id=exclude_session_id,
        )
        client = vector_store_manager.get_qdrant_client()
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=CHAT_HISTORY_COLLECTION_NAME,
                query=vector,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=limit,
            )
            return list(getattr(response, "points", []) or [])

        # Backward compatibility with older client APIs.
        return client.search(
            collection_name=CHAT_HISTORY_COLLECTION_NAME,
            query_vector=vector,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )

    def _build_search_filter(
        self,
        *,
        client_id: str,
        exclude_session_id: str | None,
    ) -> qdrant_models.Filter:
        must_not_conditions = (
            [self._field_condition("session_id", exclude_session_id)]
            if exclude_session_id
            else None
        )
        return qdrant_models.Filter(
            must=[self._field_condition("client_id", client_id)],
            must_not=must_not_conditions,
        )

    def _delete_by_field(self, *, field_name: str, value: str) -> None:
        vector_store_manager.get_qdrant_client().delete(
            collection_name=CHAT_HISTORY_COLLECTION_NAME,
            points_selector=qdrant_models.Filter(
                must=[self._field_condition(field_name, value)]
            ),
            wait=False,
        )

    @staticmethod
    def _field_condition(field_name: str, value: str) -> qdrant_models.FieldCondition:
        return qdrant_models.FieldCondition(
            key=field_name,
            match=qdrant_models.MatchValue(value=value),
        )

    @staticmethod
    def _point_to_match(point: Any) -> ChatHistoryMatch | None:
        payload = getattr(point, "payload", None) or {}
        match = ChatHistoryMatch(
            score=float(getattr(point, "score", 0.0) or 0.0),
            client_id=str(payload.get("client_id") or ""),
            session_id=str(payload.get("session_id") or ""),
            user_text=str(payload.get("user_text") or ""),
            assistant_text=str(payload.get("assistant_text") or ""),
            assistant_message_id=(
                str(payload.get("assistant_message_id"))
                if payload.get("assistant_message_id")
                else None
            ),
            created_at=(
                str(payload.get("created_at")) if payload.get("created_at") else None
            ),
        )
        if not match.user_text or not match.assistant_text:
            return None
        return match

    def _ensure_collection(self) -> None:
        if self._collection_checked:
            return

        client = vector_store_manager.get_qdrant_client()
        if not client.collection_exists(CHAT_HISTORY_COLLECTION_NAME):
            client.create_collection(
                collection_name=CHAT_HISTORY_COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=VECTOR_DIMENSIONS,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            client.create_payload_index(
                collection_name=CHAT_HISTORY_COLLECTION_NAME,
                field_name="client_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            client.create_payload_index(
                collection_name=CHAT_HISTORY_COLLECTION_NAME,
                field_name="session_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            logger.info(
                "Created chat history collection '%s'", CHAT_HISTORY_COLLECTION_NAME
            )
            self._collection_checked = True
            return

        collection_info = client.get_collection(CHAT_HISTORY_COLLECTION_NAME)
        vectors_config = collection_info.config.params.vectors
        configured_size: int | None = None
        if isinstance(vectors_config, dict):
            first_value = next(iter(vectors_config.values()), None)
            configured_size = getattr(first_value, "size", None)
        else:
            configured_size = getattr(vectors_config, "size", None)
        if configured_size is not None and configured_size != VECTOR_DIMENSIONS:
            raise RuntimeError(
                "Collection '%s' has vector size %s but expected %s"
                % (CHAT_HISTORY_COLLECTION_NAME, configured_size, VECTOR_DIMENSIONS)
            )
        self._collection_checked = True

    def _embed_text(self, text: str) -> list[float] | None:
        model = self._get_embed_model()
        embedding = model.get_text_embedding(text)
        if not embedding:
            return None
        return [float(value) for value in embedding]

    def _get_embed_model(self):
        if self._embed_model is None:
            self._embed_model = get_embeddings()
        return self._embed_model


chat_history_store = ChatHistoryVectorStore()
