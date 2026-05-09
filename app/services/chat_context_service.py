import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models.chat import ChatMessage, ChatSession
from app.indexing.chat_history_store import ChatHistoryMatch, chat_history_store


@dataclass(frozen=True)
class ChatContextBundle:
    session_summary: str
    recent_turns: list[dict[str, str]]
    cross_session_pairs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_summary": self.session_summary,
            "recent_turns": self.recent_turns,
            "cross_session_pairs": self.cross_session_pairs,
        }


class ChatContextService:
    """Build and maintain conversation context for multi-session memory."""

    def __init__(self, db: Session):
        self.db = db

    def build_context_bundle(
        self,
        *,
        client_id: str,
        session_id: str,
        current_question: str,
    ) -> ChatContextBundle:
        session_summary = self._get_session_summary(session_id)
        recent_turns = self._get_recent_turns(
            session_id=session_id,
            current_question=current_question,
            limit=settings.CHAT_SESSION_RECENT_TURNS,
        )
        cross_session_pairs = self._get_cross_session_pairs(
            client_id=client_id,
            session_id=session_id,
            question=current_question,
            limit=settings.CHAT_CROSS_SESSION_TOP_K,
        )
        return ChatContextBundle(
            session_summary=session_summary,
            recent_turns=recent_turns,
            cross_session_pairs=cross_session_pairs,
        )

    def index_qa_pair(
        self,
        *,
        client_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str,
        assistant_message_id: str,
        created_at,
    ) -> None:
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{client_id}:{session_id}:{assistant_message_id}",
            )
        )
        chat_history_store.index_qa_pair(
            point_id=point_id,
            client_id=client_id,
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            assistant_message_id=assistant_message_id,
            created_at=created_at,
        )

    def delete_session_memory(self, session_id: str) -> None:
        chat_history_store.delete_session(session_id)

    def delete_client_memory(self, client_id: str) -> None:
        chat_history_store.delete_client(client_id)

    def _get_session_summary(self, session_id: str) -> str:
        session = (
            self.db.query(ChatSession).filter(ChatSession.id == session_id).first()
        )
        if not session:
            return ""
        return (session.summary_text or "").strip()

    def _get_recent_turns(
        self,
        *,
        session_id: str,
        current_question: str,
        limit: int,
    ) -> list[dict[str, str]]:
        rows = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.turn_index.desc(), ChatMessage.created_at.desc())
            .limit(max(1, limit + 1))
            .all()
        )
        rows = list(reversed(rows))
        turns = [
            {
                "role": str(row.role),
                "content": str(row.content),
            }
            for row in rows
            if row.content
        ]
        if (
            turns
            and turns[-1]["role"] == "user"
            and turns[-1]["content"].strip() == current_question.strip()
        ):
            turns = turns[:-1]
        return turns[-limit:]

    def _get_cross_session_pairs(
        self,
        *,
        client_id: str,
        session_id: str,
        question: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        matches: list[ChatHistoryMatch] = chat_history_store.search(
            client_id=client_id,
            query_text=question,
            limit=limit,
            exclude_session_id=session_id,
        )
        return [
            {
                "session_id": match.session_id,
                "score": match.score,
                "user_text": match.user_text,
                "assistant_text": match.assistant_text,
                "assistant_message_id": match.assistant_message_id,
                "created_at": match.created_at,
            }
            for match in matches
        ]
