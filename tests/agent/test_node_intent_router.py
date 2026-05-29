"""Tests for intent_router_node."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.agent.helpers import make_state


def _mock_llm_response(route: str) -> MagicMock:
    response = MagicMock()
    response.message.content = f'{{"route": "{route}", "reason": "test reason"}}'
    return response


def test_internal_route(monkeypatch):
    monkeypatch.setattr(
        "app.agents.nodes.intent_router.invoke_llm_chat",
        lambda **kw: _mock_llm_response("internal"),
    )
    from app.agents.nodes.intent_router import intent_router_node

    result = intent_router_node(make_state(question="What is BXP FFO per share?"))
    assert result["route"] == "internal"
    assert "intent_labels" in result


def test_hybrid_route(monkeypatch):
    monkeypatch.setattr(
        "app.agents.nodes.intent_router.invoke_llm_chat",
        lambda **kw: _mock_llm_response("hybrid"),
    )
    from app.agents.nodes.intent_router import intent_router_node

    result = intent_router_node(make_state(question="How does BXP compare to sector cap rates?"))
    assert result["route"] == "hybrid"


def test_fallback_on_llm_error(monkeypatch):
    def raise_error(**kw):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr("app.agents.nodes.intent_router.invoke_llm_chat", raise_error)
    from app.agents.nodes.intent_router import intent_router_node

    result = intent_router_node(make_state())
    assert result["route"] == "internal"


def test_fallback_on_bad_json(monkeypatch):
    bad = MagicMock()
    bad.message.content = "not json at all"
    monkeypatch.setattr(
        "app.agents.nodes.intent_router.invoke_llm_chat",
        lambda **kw: bad,
    )
    from app.agents.nodes.intent_router import intent_router_node

    result = intent_router_node(make_state())
    assert result["route"] == "internal"


def test_status_callback_called(monkeypatch):
    monkeypatch.setattr(
        "app.agents.nodes.intent_router.invoke_llm_chat",
        lambda **kw: _mock_llm_response("internal"),
    )
    from app.agents.nodes.intent_router import intent_router_node

    calls = []
    result = intent_router_node(make_state(status_callback=calls.append))
    assert any("intent" in c.lower() for c in calls)
