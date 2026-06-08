from __future__ import annotations

import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.ai_provider import invoke_llm_chat
from app.core.message_manager import extract_chat_response_text
from app.core.config import settings
from app.core.prompts import SESSION_SUMMARY_DEVELOPER_PROMPT
from app.core.token_budget import ResponsesInputBudgeter, truncate_text_by_tokens
from app.db.models.chat import ChatMessage, ChatSession
from app.services.chat_context_service import ChatContextService
from app.services.query_service import execute_query

logger = logging.getLogger(__name__)


class ChatConversationService:
    """Session-aware query orchestration with client-scoped shared memory."""

    def __init__(self, db: Session):
        self.db = db
        self.context_service = ChatContextService(db)

    def execute_client_query(
        self,
        *,
        client_id: str,
        question: str,
        llm_model: str | None = None,
        reasoning_effort: str = "medium",
        reasoning_summary: str | None = None,
        session_id: str | None = None,
        status_callback=None,
        reasoning_callback=None,
        answer_callback=None,
        session_callback=None,
        skip_conversation_context: bool = False,
    ) -> dict:
        session = self._resolve_session(client_id=client_id, session_id=session_id)
        next_turn_index = self._next_turn_index(session.id)

        user_message = self._create_message(
            client_id=client_id,
            session_id=session.id,
            role="user",
            content=question,
            turn_index=next_turn_index,
        )
        self._touch_session(session, first_user_prompt=question)
        self.db.commit()
        if session_callback is not None:
            try:
                session_callback(
                    {
                        "session_id": session.id,
                        "user_message_id": user_message.id,
                    }
                )
            except Exception:
                logger.exception("Session callback failed for session %s", session.id)

        if skip_conversation_context:
            conversation_context_dict: dict = {}
        else:
            context_bundle = self.context_service.build_context_bundle(
                client_id=client_id,
                session_id=session.id,
                current_question=question,
            )
            conversation_context_dict = context_bundle.to_dict()

        try:
            result = execute_query(
                question=question,
                client_id=client_id,
                db=self.db,
                llm_model=llm_model,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                session_id=session.id,
                conversation_context=conversation_context_dict,
                status_callback=status_callback,
                reasoning_callback=reasoning_callback,
                answer_callback=answer_callback,
            )
        except Exception as exc:
            self.db.rollback()
            error_msg = f"I could not complete that search: {exc}"
            self._persist_error_message(client_id, session.id, error_msg)
            raise ValueError(error_msg) from exc

        assistant_message = self._create_message(
            client_id=client_id,
            session_id=session.id,
            role="assistant",
            content=result.get("answer", "No answer generated."),
            reasoning=result.get("reasoning"),
            citations=result.get("citations", []),
            turn_index=next_turn_index + 1,
            query_log_id=result.get("query_id"),
        )
        self._touch_session(session)
        self.context_service.index_qa_pair(
            client_id=client_id,
            session_id=session.id,
            user_text=user_message.content,
            assistant_text=assistant_message.content,
            assistant_message_id=assistant_message.id,
            created_at=assistant_message.created_at,
        )
        self._refresh_session_summary(session)
        self.db.commit()

        result.update(
            {
                "session_id": session.id,
                "user_message_id": user_message.id,
                "assistant_message_id": assistant_message.id,
            }
        )
        return result

    def list_sessions(self, *, client_id: str, limit: int = 50) -> list[ChatSession]:
        bounded_limit = max(1, min(limit, 200))
        return (
            self.db.query(ChatSession)
            .filter(ChatSession.client_id == client_id)
            .order_by(
                ChatSession.last_activity_at.desc(), ChatSession.created_at.desc()
            )
            .limit(bounded_limit)
            .all()
        )

    def create_session(
        self, *, client_id: str, title: str | None = None
    ) -> ChatSession:
        now = datetime.now(timezone.utc)
        session = ChatSession(
            id=str(uuid.uuid4()),
            client_id=client_id,
            title=(title or "").strip() or None,
            summary_text="",
            created_at=now,
            updated_at=now,
            last_activity_at=now,
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def require_session(self, *, session_id: str, client_id: str) -> ChatSession:
        session = (
            self.db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.client_id == client_id)
            .first()
        )
        if not session:
            raise ValueError(f"Session {session_id} not found for client {client_id}.")
        return session

    def list_messages(self, *, session_id: str, limit: int = 200) -> list[ChatMessage]:
        bounded_limit = max(1, min(limit, 500))
        rows = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.turn_index.desc(), ChatMessage.created_at.desc())
            .limit(bounded_limit)
            .all()
        )
        return list(reversed(rows))

    def clear_session(self, *, session_id: str, client_id: str) -> None:
        session = (
            self.db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.client_id == client_id)
            .first()
        )
        if not session:
            raise ValueError(f"Session {session_id} not found.")
        self.db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete(
            synchronize_session=False
        )
        session.summary_text = ""
        session.updated_at = datetime.now(timezone.utc)
        session.last_activity_at = datetime.now(timezone.utc)
        self.context_service.delete_session_memory(session_id)
        self.db.commit()

    def delete_session(self, *, session_id: str, client_id: str) -> None:
        """Clear all messages/memory for a session and delete the session row atomically."""
        session = (
            self.db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.client_id == client_id)
            .first()
        )
        if not session:
            raise ValueError(f"Session {session_id} not found.")
        self.db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete(
            synchronize_session=False
        )
        self.context_service.delete_session_memory(session_id)
        self.db.delete(session)
        self.db.commit()

    def _resolve_session(
        self, *, client_id: str, session_id: str | None
    ) -> ChatSession:
        if session_id:
            session = (
                self.db.query(ChatSession)
                .filter(
                    ChatSession.id == session_id,
                    ChatSession.client_id == client_id,
                )
                .first()
            )
            if not session:
                raise ValueError("Session not found for this client.")
            return session
        return self.create_session(client_id=client_id)

    def _next_turn_index(self, session_id: str) -> int:
        max_turn_index = (
            self.db.query(func.max(ChatMessage.turn_index))
            .filter(ChatMessage.session_id == session_id)
            .scalar()
        )
        return int(max_turn_index or 0) + 1

    def _create_message(
        self,
        *,
        client_id: str,
        session_id: str,
        role: str,
        content: str,
        turn_index: int,
        query_log_id: str | None = None,
        reasoning: str | None = None,
        citations: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        citations_json = None
        if citations is not None:
            normalized_citations = citations if isinstance(citations, list) else []
            try:
                citations_json = json.dumps(
                    normalized_citations, ensure_ascii=False, default=str
                )
            except Exception:
                citations_json = "[]"

        message = ChatMessage(
            id=str(uuid.uuid4()),
            client_id=client_id,
            session_id=session_id,
            role=role,
            content=content,
            reasoning=reasoning,
            citations_json=citations_json,
            turn_index=turn_index,
            query_log_id=query_log_id,
        )
        self.db.add(message)
        self.db.flush()
        return message

    def _touch_session(
        self, session: ChatSession, first_user_prompt: str | None = None
    ) -> None:
        now = datetime.now(timezone.utc)
        session.updated_at = now
        session.last_activity_at = now
        if not session.title and first_user_prompt:
            cleaned = " ".join(first_user_prompt.split())
            session.title = cleaned[:80].rstrip() if cleaned else None

    def _refresh_session_summary(self, session: ChatSession) -> None:
        rows = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session.id)
            .order_by(ChatMessage.turn_index.desc(), ChatMessage.created_at.desc())
            .limit(12)
            .all()
        )
        rows = list(reversed(rows))
        if not rows:
            session.summary_text = ""
            return

        prior_summary = (session.summary_text or "").strip()
        try:
            prompt = _build_summary_prompt(prior_summary, rows)
            model = settings.SESSION_SUMMARY_MODEL or settings.QUERY_EXPANSION_MODEL
            budgeter = ResponsesInputBudgeter(model=model)
            max_input_tokens = max(128, budgeter.input_budget_tokens)
            summary_messages = [
                {"role": "developer", "content": SESSION_SUMMARY_DEVELOPER_PROMPT},
                {"role": "user", "content": prompt},
            ]
            token_count = budgeter.count_messages_tokens(summary_messages)
            if token_count > max_input_tokens:
                fixed_tokens = budgeter.count_messages_tokens(summary_messages[:1])
                allowed_user_tokens = max(64, max_input_tokens - fixed_tokens)
                prompt = budgeter.truncate_to_tokens(prompt, allowed_user_tokens)
            response = invoke_llm_chat(
                model=model,
                input_messages=[
                    {"role": "developer", "content": SESSION_SUMMARY_DEVELOPER_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                reasoning_effort="low",
                max_output_tokens=settings.CHAT_SUMMARY_MAX_OUTPUT_TOKENS,
                prompt_cache_key="vectera:session-summary:v1",
                prompt_cache_retention=settings.RESPONSE_PROMPT_CACHE_RETENTION,
                safety_identifier=(
                    f"{settings.RESPONSE_SAFETY_IDENTIFIER_PREFIX}:{session.client_id}"
                ),
                user_tag=settings.RESPONSE_USER_TAG,
                timeout_seconds=settings.SESSION_SUMMARY_TIMEOUT_SECONDS,
            )
            candidate = extract_chat_response_text(response)
            cleaned = _sanitize_summary(candidate)
            if cleaned:
                session.summary_text = truncate_text_by_tokens(
                    cleaned,
                    max_tokens=settings.CHAT_SUMMARY_MAX_OUTPUT_TOKENS,
                    encoding_name=settings.TOKEN_BUDGET_ENCODING,
                )
                return
            raise ValueError("Summary response was empty")
        except Exception:
            logger.exception(
                "Falling back to heuristic chat summary for session %s", session.id
            )
            session.summary_text = _heuristic_summary(rows)

    def _persist_error_message(
        self, client_id: str, session_id: str, error_msg: str
    ) -> None:
        try:
            session = (
                self.db.query(ChatSession)
                .filter(
                    ChatSession.id == session_id, ChatSession.client_id == client_id
                )
                .first()
            )
            if session is not None:
                self._create_message(
                    client_id=client_id,
                    session_id=session.id,
                    role="assistant",
                    content=error_msg,
                    turn_index=self._next_turn_index(session.id),
                )
                self._touch_session(session)
                self._refresh_session_summary(session)
                self.db.commit()
        except Exception:
            self.db.rollback()
            logger.exception(
                "Failed to persist assistant error message for session %s",
                session_id,
            )


def _build_summary_prompt(prior_summary: str, rows: list[ChatMessage]) -> str:
    formatted_turns: list[str] = []
    for row in rows:
        role = "User" if row.role == "user" else "Assistant"
        content = " ".join((row.content or "").split())
        if len(content) > 400:
            content = f"{content[:397].rstrip()}..."
        formatted_turns.append(f"{role}: {content}")

    turns_block = "\n".join(formatted_turns)
    return (
        "Update the running summary for this financial analyst conversation.\n"
        "Preserve: company names, tickers, document names, fiscal periods, metrics asked about "
        "(FFO, NOI, Occupancy, WALT, Cap Rate, etc.), key conclusions reached, and open questions.\n"
        "Drop: stale or resolved context, pleasantries, and verbatim assistant responses.\n"
        "Do not repeat what the assistant said verbatim. Do not invent facts.\n"
        "If there is nothing new to add, return the existing summary unchanged.\n"
        "Output plain text only, no markdown, max 8 lines.\n\n"
        f"Existing summary:\n{prior_summary or '(none)'}\n\n"
        f"Recent turns:\n{turns_block}\n\n"
        "Updated summary:"
    )


def _sanitize_summary(text: str) -> str:
    summary = text.strip()
    if not summary:
        return ""
    # Response-mode models may include tagged sections; keep only surface summary.
    summary = summary.replace("<thinking>", "").replace("</thinking>", "")
    summary = summary.replace("<answer>", "").replace("</answer>", "")
    return summary.strip()


def _heuristic_summary(rows: list[ChatMessage]) -> str:
    tail = rows[-8:]
    lines: list[str] = []
    for row in tail:
        role = "User" if row.role == "user" else "Assistant"
        text = " ".join((row.content or "").split())
        if len(text) > 180:
            text = f"{text[:177].rstrip()}..."
        lines.append(f"{role}: {text}")
    return truncate_text_by_tokens(
        "\n".join(lines),
        max_tokens=settings.CHAT_SUMMARY_MAX_OUTPUT_TOKENS,
        encoding_name=settings.TOKEN_BUDGET_ENCODING,
    )
