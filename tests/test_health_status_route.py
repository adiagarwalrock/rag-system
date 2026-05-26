"""
Integration-style tests for the GET /health/status route.

Uses FastAPI's TestClient with RuntimeStatusService mocked out so no real
Qdrant, database, or OpenAI connections are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes_health import router


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


FAKE_STATUS = {
    "overall": {
        "parsers": "ok",
        "database": "ok",
        "qdrant": "ok",
        "ai": "ok",
    },
    "parsers": {
        "status": "ok",
        "external_enabled": False,
        "parsers": [],
        "routing_priority": ["Layout-aware PDF", "Legacy"],
    },
    "database": {
        "status": "ok",
        "mode": "local_sqlite",
        "dialect": "sqlite",
        "target": {"path": "./rag_local.db"},
        "message": "Connection check succeeded.",
    },
    "qdrant": {
        "status": "ok",
        "url": "http://localhost:6333",
        "document_collection": "docs",
        "chat_history_collection": "chat_history_v1",
        "collections": [
            {"name": "docs", "point_count": 42, "role": "documents"},
            {"name": "chat_history_v1", "point_count": 10, "role": "chat_history"},
        ],
        "message": "Collection list loaded.",
    },
    "ai": {
        "status": "ok",
        "provider": "openai",
        "llm_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
        "key_valid": True,
    },
}


def test_health_check_still_works(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "service" in body


def test_runtime_status_returns_full_payload(client):
    with patch(
        "app.api.routes_health.RuntimeStatusService"
    ) as MockService:
        MockService.return_value.get_status.return_value = FAKE_STATUS
        response = client.get("/status")

    assert response.status_code == 200
    body = response.json()
    assert "overall" in body
    assert "parsers" in body
    assert "database" in body
    assert "qdrant" in body
    assert "ai" in body


def test_runtime_status_qdrant_collections_present(client):
    with patch(
        "app.api.routes_health.RuntimeStatusService"
    ) as MockService:
        MockService.return_value.get_status.return_value = FAKE_STATUS
        response = client.get("/status")

    collections = response.json()["qdrant"]["collections"]
    assert isinstance(collections, list)
    assert len(collections) == 2
    names = {c["name"] for c in collections}
    assert "docs" in names
    assert "chat_history_v1" in names


def test_runtime_status_overall_keys(client):
    with patch(
        "app.api.routes_health.RuntimeStatusService"
    ) as MockService:
        MockService.return_value.get_status.return_value = FAKE_STATUS
        response = client.get("/status")

    overall = response.json()["overall"]
    for key in ("parsers", "database", "qdrant", "ai"):
        assert key in overall, f"Missing overall key: {key}"


def test_runtime_status_service_error_propagates():
    """If RuntimeStatusService raises, FastAPI should return a 500."""
    app = FastAPI()
    app.include_router(router)
    # raise_server_exceptions=False lets the test client return the HTTP 500
    # instead of re-raising the exception in test code.
    error_client = TestClient(app, raise_server_exceptions=False)
    with patch(
        "app.api.routes_health.RuntimeStatusService"
    ) as MockService:
        MockService.return_value.get_status.side_effect = RuntimeError("boom")
        response = error_client.get("/status")

    assert response.status_code == 500
