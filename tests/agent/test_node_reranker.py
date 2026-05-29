"""Tests for reranker_node."""

from __future__ import annotations

from tests.agent.helpers import make_node, make_state


def test_reranker_calls_rerank_and_select(monkeypatch):
    node = make_node("n1")
    ranked = [node]
    evidence = [node]

    monkeypatch.setattr(
        "app.agents.nodes.reranker.rerank_nodes",
        lambda nodes, top_k, prefer_latest, query: ranked,
    )
    monkeypatch.setattr(
        "app.agents.nodes.reranker.VecteraRetriever._select_evidence_nodes",
        lambda self, q, nodes: evidence,
    )
    from app.agents.nodes.reranker import reranker_node

    state = make_state(retrieved_nodes=[node])
    result = reranker_node(state)
    assert result["evidence_nodes"] == evidence


def test_reranker_empty_input(monkeypatch):
    monkeypatch.setattr(
        "app.agents.nodes.reranker.rerank_nodes",
        lambda nodes, **kw: [],
    )
    monkeypatch.setattr(
        "app.agents.nodes.reranker.VecteraRetriever._select_evidence_nodes",
        lambda self, q, nodes: [],
    )
    from app.agents.nodes.reranker import reranker_node

    result = reranker_node(make_state(retrieved_nodes=[]))
    assert result["evidence_nodes"] == []


def test_reranker_accumulates_across_iterations(monkeypatch):
    """Nodes from multiple iterations are all passed to rerank together."""
    n1 = make_node("n1", score=0.9)
    n2 = make_node("n2", score=0.7)
    seen_nodes = []

    def fake_rerank(nodes, **kw):
        seen_nodes.extend(nodes)
        return nodes

    monkeypatch.setattr("app.agents.nodes.reranker.rerank_nodes", fake_rerank)
    monkeypatch.setattr(
        "app.agents.nodes.reranker.VecteraRetriever._select_evidence_nodes",
        lambda self, q, nodes: nodes,
    )
    from app.agents.nodes.reranker import reranker_node

    # Simulates two retrieval iterations having accumulated both nodes
    state = make_state(retrieved_nodes=[n1, n2])
    reranker_node(state)
    assert len(seen_nodes) == 2
    assert {n.node.node_id for n in seen_nodes} == {"n1", "n2"}
