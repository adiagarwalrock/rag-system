"""Tests for vector_retrieval_node."""

from __future__ import annotations

from tests.agent.helpers import make_node, make_state


def test_returns_fresh_nodes_and_increments_count(monkeypatch):
    node = make_node("n1")
    monkeypatch.setattr(
        "app.agents.nodes.vector_retrieval.VecteraRetriever.retrieve_only",
        lambda self, q: [node],
    )
    from app.agents.nodes.vector_retrieval import vector_retrieval_node

    result = vector_retrieval_node(make_state())
    assert result["retrieved_nodes"] == [node]
    assert result["retrieved_node_ids"] == ["n1"]
    assert result["iteration_count"] == 1


def test_excludes_already_seen_nodes(monkeypatch):
    node_new = make_node("n2")
    node_old = make_node("n1")
    monkeypatch.setattr(
        "app.agents.nodes.vector_retrieval.VecteraRetriever.retrieve_only",
        lambda self, q: [node_old, node_new],
    )
    from app.agents.nodes.vector_retrieval import vector_retrieval_node

    state = make_state(retrieved_node_ids=["n1"], iteration_count=1)
    result = vector_retrieval_node(state)
    # n1 is already seen — only n2 should be returned
    assert result["retrieved_nodes"] == [node_new]
    assert result["retrieved_node_ids"] == ["n2"]


def test_uses_retrieval_gap_on_second_iteration(monkeypatch):
    captured = {}

    def fake_retrieve(self, query):
        captured["query"] = query
        return []

    monkeypatch.setattr(
        "app.agents.nodes.vector_retrieval.VecteraRetriever.retrieve_only",
        fake_retrieve,
    )
    from app.agents.nodes.vector_retrieval import vector_retrieval_node

    state = make_state(
        iteration_count=1,
        retrieval_gap="BXP cap rate breakdown Q4 2024",
    )
    vector_retrieval_node(state)
    assert captured["query"] == "BXP cap rate breakdown Q4 2024"


def test_uses_original_question_on_first_iteration(monkeypatch):
    captured = {}

    def fake_retrieve(self, query):
        captured["query"] = query
        return []

    monkeypatch.setattr(
        "app.agents.nodes.vector_retrieval.VecteraRetriever.retrieve_only",
        fake_retrieve,
    )
    from app.agents.nodes.vector_retrieval import vector_retrieval_node

    state = make_state(question="What is BXP FFO?", iteration_count=0, retrieval_gap=None)
    vector_retrieval_node(state)
    assert captured["query"] == "What is BXP FFO?"
