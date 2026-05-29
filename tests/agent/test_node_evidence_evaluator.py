"""Tests for evidence_evaluator_node."""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.agent.helpers import make_node, make_state


def _mock_llm(sufficient: bool, gap: str | None = None) -> MagicMock:
    import json
    response = MagicMock()
    response.message.content = json.dumps({"sufficient": sufficient, "gap": gap})
    return response


def test_force_sufficient_at_max_iterations(monkeypatch):
    monkeypatch.setattr("app.agents.nodes.evidence_evaluator.settings.AGENTIC_MAX_ITERATIONS", 5)
    from app.agents.nodes.evidence_evaluator import evidence_evaluator_node

    state = make_state(iteration_count=5, retrieved_nodes=[make_node()])
    result = evidence_evaluator_node(state)
    assert result["evidence_sufficient"] is True
    assert result["retrieval_gap"] is None


def test_insufficient_with_gap(monkeypatch):
    monkeypatch.setattr("app.agents.nodes.evidence_evaluator.settings.AGENTIC_MAX_ITERATIONS", 5)
    monkeypatch.setattr(
        "app.agents.nodes.evidence_evaluator.invoke_llm_chat",
        lambda **kw: _mock_llm(False, "missing Q4 2024 guidance data"),
    )
    from app.agents.nodes.evidence_evaluator import evidence_evaluator_node

    state = make_state(iteration_count=1, retrieved_nodes=[make_node()])
    result = evidence_evaluator_node(state)
    assert result["evidence_sufficient"] is False
    assert result["retrieval_gap"] == "missing Q4 2024 guidance data"


def test_sufficient(monkeypatch):
    monkeypatch.setattr("app.agents.nodes.evidence_evaluator.settings.AGENTIC_MAX_ITERATIONS", 5)
    monkeypatch.setattr(
        "app.agents.nodes.evidence_evaluator.invoke_llm_chat",
        lambda **kw: _mock_llm(True),
    )
    from app.agents.nodes.evidence_evaluator import evidence_evaluator_node

    state = make_state(iteration_count=1, retrieved_nodes=[make_node()])
    result = evidence_evaluator_node(state)
    assert result["evidence_sufficient"] is True


def test_empty_nodes_returns_insufficient(monkeypatch):
    monkeypatch.setattr("app.agents.nodes.evidence_evaluator.settings.AGENTIC_MAX_ITERATIONS", 5)
    from app.agents.nodes.evidence_evaluator import evidence_evaluator_node

    state = make_state(iteration_count=0, retrieved_nodes=[])
    result = evidence_evaluator_node(state)
    assert result["evidence_sufficient"] is False
    assert result["retrieval_gap"] is not None


def test_fallback_to_sufficient_on_llm_error(monkeypatch):
    monkeypatch.setattr("app.agents.nodes.evidence_evaluator.settings.AGENTIC_MAX_ITERATIONS", 5)

    def raise_error(**kw):
        raise RuntimeError("timeout")

    monkeypatch.setattr("app.agents.nodes.evidence_evaluator.invoke_llm_chat", raise_error)
    from app.agents.nodes.evidence_evaluator import evidence_evaluator_node

    state = make_state(iteration_count=1, retrieved_nodes=[make_node()])
    result = evidence_evaluator_node(state)
    assert result["evidence_sufficient"] is True
