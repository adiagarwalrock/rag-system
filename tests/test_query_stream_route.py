"""
Tests for the POST /query/ route — both non-streaming (stream: false) and
streaming (stream: true / SSE) paths.

All external dependencies (DB, ChatConversationService, ClientLookupService)
are mocked so no real Qdrant, OpenAI, or SQLAlchemy connection is required.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes_query import router

FAKE_RESULT = {
    "answer": "There are 17 indulgences.",
    "reasoning": "I looked at the table on page 14…",
    "citations": [{"filename": "vici_annual.pdf", "page": 14}],
    "conflicts": [],
    "query_id": "qid-123",
    "latency_ms": 1234,
    "source_count": 3,
    "evidence_count": 5,
    "images_used": [],
    "image_evidence_count": 0,
    "reasoning_effort": "medium",
    "reasoning_effort_applied": True,
    "session_id": "sess-abc",
    "user_message_id": "umsg-1",
    "assistant_message_id": "amsg-2",
}

BASE_PAYLOAD = {
    "question": "How many indulgences?",
    "client_id": "client-xyz",
    "reasoning_effort": "medium",
}


@pytest.fixture()
def client(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    monkeypatch.setattr("app.api.routes_query.SessionLocal", lambda: MagicMock())
    return TestClient(app, raise_server_exceptions=False)


def _mock_service(
    status_callback=None,
    reasoning_callback=None,
    answer_callback=None,
    session_callback=None,
    **_kwargs,
):
    """Simulate the service emitting phase events then returning FAKE_RESULT."""
    if session_callback:
        session_callback(
            {
                "session_id": FAKE_RESULT["session_id"],
                "user_message_id": FAKE_RESULT["user_message_id"],
            }
        )
    if status_callback:
        status_callback("🔍 Retrieving relevant chunks…")
        status_callback("📊 Reranking 10 candidate chunks…")
        status_callback("✂️ Selecting evidence…")
        status_callback("🧠 Synthesizing grounded answer…")
    if reasoning_callback:
        reasoning_callback("I looked ")
        reasoning_callback("at the table…")
    if answer_callback:
        answer_callback("There are ")
        answer_callback("17 indulgences.")
    return FAKE_RESULT


def _mock_silent_service(session_callback=None, **_kwargs):
    if session_callback:
        session_callback(
            {
                "session_id": FAKE_RESULT["session_id"],
                "user_message_id": FAKE_RESULT["user_message_id"],
            }
        )
    return FAKE_RESULT


# ── Non-streaming path ────────────────────────────────────────────────────────

def test_non_streaming_returns_json(client):
    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.return_value = FAKE_RESULT

        response = client.post("/", json={**BASE_PAYLOAD, "stream": False})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["answer"] == FAKE_RESULT["answer"]
    assert body["query_id"] == FAKE_RESULT["query_id"]


def test_non_streaming_default_stream_false(client):
    """Omitting 'stream' should behave identically to stream: false."""
    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.return_value = FAKE_RESULT

        response = client.post("/", json=BASE_PAYLOAD)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


def test_non_streaming_client_not_found(client):
    with patch("app.api.routes_query.ClientLookupService") as MockLookup:
        MockLookup.return_value.require_client.side_effect = ValueError("not found")
        response = client.post("/", json=BASE_PAYLOAD)

    assert response.status_code == 404


def test_non_streaming_does_not_pass_callbacks(client):
    """Callbacks must be None on the non-streaming path."""
    captured: dict = {}

    def capture_call(**kwargs):
        captured.update(kwargs)
        return FAKE_RESULT

    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.side_effect = capture_call
        client.post("/", json=BASE_PAYLOAD)

    assert captured.get("status_callback") is None
    assert captured.get("reasoning_callback") is None


# ── Streaming path ────────────────────────────────────────────────────────────

def _collect_sse_events(response) -> list[dict]:
    """Parse raw SSE text into a list of dicts with 'type' and 'data' keys.

    FastAPI's EventSourceResponse emits named SSE events in the form:
        event: <type>
        data: <json>

    Each parsed entry has shape: {"type": str, "data": dict}.
    """
    events = []
    current_event_type: str | None = None
    for raw_line in response.text.splitlines():
        line = raw_line.strip()
        if line.startswith("event:"):
            current_event_type = line[6:].strip()
        elif line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                events.append({
                    "type": current_event_type,
                    "data": json.loads(payload),
                })
            current_event_type = None
        elif not line:
            current_event_type = None
    return events


def test_streaming_content_type(client):
    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.side_effect = _mock_service

        response = client.post("/", json={**BASE_PAYLOAD, "stream": True})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_streaming_emits_status_events(client):
    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.side_effect = _mock_service

        response = client.post("/", json={**BASE_PAYLOAD, "stream": True})

    events = _collect_sse_events(response)
    status_events = [e for e in events if e.get("type") == "status"]
    assert len(status_events) >= 1
    assert any("Retrieving" in e["data"].get("delta", "") for e in status_events)


def test_streaming_emits_final_without_status_when_service_is_silent(client):
    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.side_effect = _mock_silent_service

        response = client.post("/", json={**BASE_PAYLOAD, "stream": True})

    events = _collect_sse_events(response)
    types = [event.get("type") for event in events]
    assert types == ["session", "final"]


def test_streaming_emits_session_before_reasoning_answer_and_final(client):
    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.side_effect = _mock_service

        response = client.post("/", json={**BASE_PAYLOAD, "stream": True})

    events = _collect_sse_events(response)
    types = [event.get("type") for event in events]
    assert types[0] == "session"
    assert types.index("session") < types.index("reasoning")
    assert types.index("session") < types.index("answer")
    assert types.index("session") < types.index("final")
    session = events[0]["data"]
    assert session["session_id"] == FAKE_RESULT["session_id"]
    assert session["user_message_id"] == FAKE_RESULT["user_message_id"]


def test_streaming_emits_reasoning_events(client):
    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.side_effect = _mock_service

        response = client.post("/", json={**BASE_PAYLOAD, "stream": True})

    events = _collect_sse_events(response)
    reasoning_events = [e for e in events if e.get("type") == "reasoning"]
    assert len(reasoning_events) >= 1


def test_streaming_emits_answer_delta_events(client):
    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.side_effect = _mock_service

        response = client.post("/", json={**BASE_PAYLOAD, "stream": True})

    events = _collect_sse_events(response)
    answer_events = [e for e in events if e.get("type") == "answer"]
    assert [event["data"].get("delta") for event in answer_events] == [
        "There are ",
        "17 indulgences.",
    ]


def test_streaming_emits_final_event(client):
    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.side_effect = _mock_service

        response = client.post("/", json={**BASE_PAYLOAD, "stream": True})

    events = _collect_sse_events(response)
    final_events = [e for e in events if e.get("type") == "final"]
    assert len(final_events) == 1
    # With named SSE events the full result dict IS the data payload.
    resp = final_events[0]["data"]
    assert resp["answer"] == FAKE_RESULT["answer"]
    assert resp["query_id"] == FAKE_RESULT["query_id"]
    assert isinstance(resp["citations"], list)


def test_non_streaming_returns_retrieval_trace(client, monkeypatch, tmp_path):
    parsed_root = tmp_path / "parsed"
    image_path = parsed_root / "doc-1" / "screenshots" / "page_1.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image")
    monkeypatch.setattr("app.retrieval.citation_builder.settings.PARSED_ARTIFACTS_DIR", str(parsed_root))

    result = {
        **FAKE_RESULT,
        "retrieval_mode": "hybrid",
        "query_expanded": True,
        "intent_labels": ["visual_lookup"],
        "companion_queries": ["show chart image"],
        "companion_counts_by_query": {"show chart image": 2},
        "images_used": [str(image_path)],
        "image_evidence_count": 1,
        "retrieval_diagnostics": {
            "ranked_image_chunk_count": 4,
            "evidence_image_chunk_count": 1,
        },
    }

    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.return_value = result

        response = client.post("/", json={**BASE_PAYLOAD, "stream": False})

    assert response.status_code == 200
    retrieval = response.json()["retrieval"]
    assert retrieval["mode"] == "hybrid"
    assert retrieval["query_expanded"] is True
    assert retrieval["image_referenced"] is True
    assert retrieval["images_used_count"] == 1
    assert retrieval["ranked_image_chunk_count"] == 4
    assert retrieval["evidence_image_chunk_count"] == 1
    assert retrieval["image_assets_used"][0]["url"] == (
        "/api/v1/artifacts/image?path=doc-1/screenshots/page_1.png"
    )
    assert retrieval["intent_labels"] == ["visual_lookup"]
    assert retrieval["companion_counts_by_query"] == {"show chart image": 2}


def test_streaming_final_event_replaces_non_finite_scores(client):
    result = {
        **FAKE_RESULT,
        "citations": [
            {
                "filename": "vici_annual.pdf",
                "page": 14,
                "score": math.nan,
                "nested": {"score": math.inf},
            }
        ],
    }

    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        def _service_with_nan(session_callback=None, **_kwargs):
            if session_callback:
                session_callback(
                    {
                        "session_id": result["session_id"],
                        "user_message_id": result["user_message_id"],
                    }
                )
            return result

        MockSvc.return_value.execute_client_query.side_effect = _service_with_nan

        response = client.post("/", json={**BASE_PAYLOAD, "stream": True})

    assert "NaN" not in response.text
    assert "Infinity" not in response.text

    events = _collect_sse_events(response)
    final_events = [e for e in events if e.get("type") == "final"]
    assert len(final_events) == 1
    citation = final_events[0]["data"]["citations"][0]
    assert citation["score"] is None
    assert citation["nested"]["score"] is None


def test_streaming_event_order(client):
    """Session and status events must precede the final event."""
    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.side_effect = _mock_service

        response = client.post("/", json={**BASE_PAYLOAD, "stream": True})

    events = _collect_sse_events(response)
    types = [e.get("type") for e in events]
    assert types[0] == "session"
    assert "final" in types
    final_index = types.index("final")
    for i, t in enumerate(types):
        if t in {"session", "status", "reasoning", "answer"}:
            assert i < final_index


def test_streaming_passes_callbacks_to_service(client):
    """The streaming path must supply non-None callbacks to execute_client_query."""
    captured: dict = {}

    def capture_call(**kwargs):
        captured.update(kwargs)
        return FAKE_RESULT

    with (
        patch("app.api.routes_query.ClientLookupService") as MockLookup,
        patch("app.api.routes_query.ChatConversationService") as MockSvc,
    ):
        MockLookup.return_value.require_client.return_value = MagicMock()
        MockSvc.return_value.execute_client_query.side_effect = capture_call
        client.post("/", json={**BASE_PAYLOAD, "stream": True})

    assert callable(captured.get("status_callback"))
    assert callable(captured.get("reasoning_callback"))
    assert callable(captured.get("answer_callback"))
    assert callable(captured.get("session_callback"))
