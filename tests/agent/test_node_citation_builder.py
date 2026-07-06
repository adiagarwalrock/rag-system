"""Tests for citation_builder_node."""

from __future__ import annotations

from tests.agent.helpers import make_node, make_state


def test_citations_built_from_evidence_nodes(monkeypatch):
    node = make_node("n1")
    fake_citation = {"rank": 1, "score": 0.9, "text": "snippet", "vector_node_id": "n1"}

    monkeypatch.setattr(
        "app.agents.nodes.citation_builder.build_citations",
        lambda nodes: [fake_citation],
    )
    from app.agents.nodes.citation_builder import citation_builder_node

    result = citation_builder_node(make_state(evidence_nodes=[node]))
    assert result["citations"] == [fake_citation]


def test_empty_evidence_returns_empty_citations(monkeypatch):
    monkeypatch.setattr(
        "app.agents.nodes.citation_builder.build_citations",
        lambda nodes: [],
    )
    from app.agents.nodes.citation_builder import citation_builder_node

    result = citation_builder_node(make_state(evidence_nodes=[]))
    assert result["citations"] == []
