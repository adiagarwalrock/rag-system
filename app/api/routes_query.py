"""
Query API route — separate from documents.

Supports two response modes controlled by the ``stream`` field in the request body:

* ``stream: false`` (default) — standard synchronous JSON response.
* ``stream: true`` — Server-Sent Events stream.  Each SSE event has a named
  ``event`` field so clients can filter by type without parsing JSON:

    event: status     data: {"delta": "<phase message>"}
    event: reasoning  data: {"delta": "<reasoning token>"}
    event: final      data: {<QueryResponse fields>}
    event: error      data: {"detail": "<error message>"}

  The streaming path runs the synchronous service layer in a daemon thread, bridges
  results through a bounded ``queue.Queue``, and drains it via an async generator
  returned as a ``text/event-stream`` ``StreamingResponse``.
"""

from __future__ import annotations

import asyncio
import json
import queue as stdlib_queue
import threading
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.snowflake import get_db
from app.schemas.document import QueryRequest, QueryResponse
from app.services.chat_conversation_service import ChatConversationService
from app.services.client_service import ClientLookupService

router = APIRouter()

# Sentinel that signals the worker thread has finished.
_DONE = object()


def _build_query_response(result: dict[str, Any]) -> QueryResponse:
    return QueryResponse(
        answer=result["answer"],
        reasoning=result.get("reasoning"),
        citations=result.get("citations", []),
        conflicts=result.get("conflicts", []),
        query_id=result.get("query_id"),
        latency_ms=result.get("latency_ms"),
        source_count=result.get("source_count", 0),
        evidence_count=result.get("evidence_count", 0),
        images_used=result.get("images_used", []),
        image_evidence_count=result.get("image_evidence_count", 0),
        reasoning_effort=result.get("reasoning_effort", "medium"),
        reasoning_effort_applied=result.get("reasoning_effort_applied", False),
        session_id=result.get("session_id"),
        user_message_id=result.get("user_message_id"),
        assistant_message_id=result.get("assistant_message_id"),
    )


def _sse_chunk(event_type: str, data: Any) -> str:
    """Format a single SSE event as a pre-encoded string chunk."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.post("/", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db),
):
    """Query documents scoped to a client.

    Set ``stream: true`` in the request body to receive a Server-Sent Events
    response with live phase status, reasoning deltas, and a final payload.
    Omit ``stream`` (or set ``stream: false``) for the standard JSON response.
    """
    try:
        ClientLookupService(db).require_client(request.client_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Client not found")

    if not request.stream:
        # ── Non-streaming path (unchanged behaviour) ──────────────────────
        result = ChatConversationService(db).execute_client_query(
            question=request.question,
            client_id=request.client_id,
            reasoning_effort=request.reasoning_effort,
            reasoning_summary=request.reasoning_summary,
            session_id=request.session_id,
        )
        return _build_query_response(result)

    # ── Streaming path ────────────────────────────────────────────────────
    # The service layer uses blocking I/O (SQLAlchemy + OpenAI SDK).  We run
    # it in a daemon thread and bridge output through a bounded queue.
    # The async generator drains the queue via ``run_in_executor`` so the
    # event loop is never blocked.
    event_queue: stdlib_queue.Queue = stdlib_queue.Queue(maxsize=256)

    def _status_cb(msg: str) -> None:
        try:
            event_queue.put_nowait(_sse_chunk("status", {"delta": msg}))
        except stdlib_queue.Full:
            pass

    def _reasoning_cb(delta: str) -> None:
        try:
            event_queue.put_nowait(_sse_chunk("reasoning", {"delta": delta}))
        except stdlib_queue.Full:
            pass

    def _run_service() -> None:
        try:
            result = ChatConversationService(db).execute_client_query(
                question=request.question,
                client_id=request.client_id,
                reasoning_effort=request.reasoning_effort,
                reasoning_summary=request.reasoning_summary,
                session_id=request.session_id,
                status_callback=_status_cb,
                reasoning_callback=_reasoning_cb,
            )
            event_queue.put(_sse_chunk("final", result))
        except Exception as exc:
            event_queue.put(_sse_chunk("error", {"detail": str(exc)}))
        finally:
            event_queue.put(_DONE)

    # Start the worker thread before returning the response so the queue is
    # already being populated when the async generator first polls it.
    threading.Thread(target=_run_service, daemon=True).start()

    async def _generate() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        while True:
            # Non-blocking poll: yields the event loop while waiting.
            raw = await loop.run_in_executor(None, event_queue.get)
            if raw is _DONE:
                break
            yield raw

    return StreamingResponse(_generate(), media_type="text/event-stream")
