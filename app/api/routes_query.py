"""
Query API route — separate from documents.

Supports two response modes controlled by the ``stream`` field in the request body:

* ``stream: false`` (default) — standard synchronous JSON response.
* ``stream: true`` — Server-Sent Events stream.  Each SSE event has a named
  ``event`` field so clients can filter by type without parsing JSON:

    event: session    data: {"session_id": "...", "user_message_id": "..."}
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
import math
import queue as stdlib_queue
import threading
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.snowflake import SessionLocal, get_db
from app.retrieval.citation_builder import build_image_assets
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
        retrieval=_build_retrieval_trace(result),
        reasoning_effort=result.get("reasoning_effort", "medium"),
        reasoning_effort_applied=result.get("reasoning_effort_applied", False),
        reasoning_summary=result.get("reasoning_summary"),
        session_id=result.get("session_id"),
        user_message_id=result.get("user_message_id"),
        assistant_message_id=result.get("assistant_message_id"),
    )


def _build_retrieval_trace(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("retrieval_diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    images_used = result.get("images_used")
    if not isinstance(images_used, list):
        images_used = []

    return {
        "mode": result.get("retrieval_mode", "dense_only"),
        "top_k": result.get("top_k"),
        "fallback_reason": result.get("fallback_reason"),
        "query_expanded": bool(result.get("query_expanded", False)),
        "intent_labels": result.get("intent_labels")
        or diagnostics.get("intent_labels")
        or [],
        "companion_queries": result.get("companion_queries", []),
        "companion_counts_by_query": result.get("companion_counts_by_query", {}),
        "image_referenced": bool(images_used),
        "image_evidence_count": result.get("image_evidence_count", 0),
        "images_used_count": len(images_used),
        "image_assets_used": build_image_assets(
            asset_refs=[str(path) for path in images_used],
            document_name="Model image evidence",
        ),
        "ranked_image_chunk_count": diagnostics.get("ranked_image_chunk_count", 0),
        "evidence_image_chunk_count": diagnostics.get("evidence_image_chunk_count", 0),
        "source_count": result.get("source_count", 0),
        "evidence_count": result.get("evidence_count", 0),
        "ranked_document_count": diagnostics.get("ranked_document_count"),
        "evidence_document_count": diagnostics.get("evidence_document_count"),
        "ranked_chunk_types": diagnostics.get("ranked_chunk_types", {}),
        "evidence_chunk_types": diagnostics.get("evidence_chunk_types", {}),
    }


def _sse_chunk(event_type: str, data: Any) -> str:
    """Format a single SSE event as a pre-encoded string chunk."""
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(_json_safe(data), allow_nan=False)}\n\n"
    )


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable value without non-standard NaN/Infinity tokens."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


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

    effort = request.reasoning_effort or settings.REASONING_EFFORT

    if not request.stream:
        # ── Non-streaming path ────────────────────────────────────────────
        result = ChatConversationService(db).execute_client_query(
            question=request.question,
            client_id=request.client_id,
            reasoning_effort=effort,
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

    def _delta_cb(event_type: str):
        def _cb(delta: str) -> None:
            try:
                event_queue.put_nowait(_sse_chunk(event_type, {"delta": delta}))
            except stdlib_queue.Full:
                pass
        return _cb

    _status_cb = _delta_cb("status")
    _reasoning_cb = _delta_cb("reasoning")
    _answer_cb = _delta_cb("answer")

    def _session_cb(payload: dict[str, Any]) -> None:
        event_queue.put(_sse_chunk("session", payload))

    def _run_service() -> None:
        worker_db = SessionLocal()
        try:
            result = ChatConversationService(worker_db).execute_client_query(
                question=request.question,
                client_id=request.client_id,
                reasoning_effort=effort,
                reasoning_summary=request.reasoning_summary,
                session_id=request.session_id,
                status_callback=_status_cb,
                reasoning_callback=_reasoning_cb,
                answer_callback=_answer_cb,
                session_callback=_session_cb,
            )
            event_queue.put(
                _sse_chunk("final", _build_query_response(result).model_dump())
            )
        except Exception as exc:
            event_queue.put(_sse_chunk("error", {"detail": str(exc)}))
        finally:
            worker_db.close()
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
