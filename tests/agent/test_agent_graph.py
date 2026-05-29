"""Integration tests for the full LangGraph agent graph."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.agent.helpers import make_node, make_state


def _patch_all_nodes(monkeypatch, *, retrieved_nodes=None, sufficient=True, gap=None):
    """Patch every node to return minimal valid state updates."""
    nodes = retrieved_nodes or [make_node()]

    monkeypatch.setattr(
        "app.agents.nodes.intent_router.invoke_llm_chat",
        lambda **kw: _llm_response("internal"),
    )
    monkeypatch.setattr(
        "app.agents.nodes.vector_retrieval.VecteraRetriever.retrieve_only",
        lambda self, q: nodes,
    )
    monkeypatch.setattr(
        "app.agents.nodes.evidence_evaluator.invoke_llm_chat",
        lambda **kw: _llm_eval(sufficient, gap),
    )
    monkeypatch.setattr(
        "app.agents.nodes.reranker.rerank_nodes",
        lambda ns, **kw: ns,
    )
    monkeypatch.setattr(
        "app.agents.nodes.reranker.VecteraRetriever._select_evidence_nodes",
        lambda self, q, ns: ns,
    )
    monkeypatch.setattr(
        "app.agents.nodes.conflict_detector.detect_conflicts",
        lambda **kw: [],
    )
    monkeypatch.setattr(
        "app.agents.nodes.citation_builder.build_citations",
        lambda ns: [{"rank": 1, "score": 0.9, "text": "t", "vector_node_id": "n1"}],
    )
    monkeypatch.setattr(
        "app.agents.nodes.synthesizer.GroundedAnswerSynthesizer.synthesize",
        lambda self, **kw: {
            "answer": "The FFO is $3.50.",
            "reasoning": None,
            "images_used": [],
            "reasoning_effort_applied": False,
        },
    )


def _llm_response(route: str) -> MagicMock:
    import json
    r = MagicMock()
    r.message.content = json.dumps({"route": route, "reason": "test"})
    return r


def _llm_eval(sufficient: bool, gap=None) -> MagicMock:
    import json
    r = MagicMock()
    r.message.content = json.dumps({"sufficient": sufficient, "gap": gap})
    return r


def test_full_graph_internal_path(monkeypatch):
    _patch_all_nodes(monkeypatch, sufficient=True)
    monkeypatch.setattr("app.agents.nodes.evidence_evaluator.settings.AGENTIC_MAX_ITERATIONS", 5)

    from app.agents.graph import build_graph

    graph = build_graph()
    final = graph.invoke(make_state())
    assert final["answer"] == "The FFO is $3.50."
    assert final["iteration_count"] == 1


def test_graph_loops_on_insufficient_then_synthesizes(monkeypatch):
    """Graph should loop once when evidence is insufficient, then synthesize."""
    call_count = {"n": 0}

    def counting_retrieve(self, q):
        call_count["n"] += 1
        return [make_node(f"n{call_count['n']}")]

    eval_count = {"n": 0}

    def alternating_eval(**kw):
        import json
        eval_count["n"] += 1
        # First eval: not sufficient. Second eval: sufficient.
        sufficient = eval_count["n"] >= 2
        r = MagicMock()
        r.message.content = json.dumps({"sufficient": sufficient, "gap": "more detail needed"})
        return r

    monkeypatch.setattr("app.agents.nodes.evidence_evaluator.settings.AGENTIC_MAX_ITERATIONS", 5)
    monkeypatch.setattr(
        "app.agents.nodes.intent_router.invoke_llm_chat",
        lambda **kw: _llm_response("internal"),
    )
    monkeypatch.setattr(
        "app.agents.nodes.vector_retrieval.VecteraRetriever.retrieve_only",
        counting_retrieve,
    )
    monkeypatch.setattr("app.agents.nodes.evidence_evaluator.invoke_llm_chat", alternating_eval)
    monkeypatch.setattr("app.agents.nodes.reranker.rerank_nodes", lambda ns, **kw: ns)
    monkeypatch.setattr(
        "app.agents.nodes.reranker.VecteraRetriever._select_evidence_nodes",
        lambda self, q, ns: ns,
    )
    monkeypatch.setattr("app.agents.nodes.conflict_detector.detect_conflicts", lambda **kw: [])
    monkeypatch.setattr(
        "app.agents.nodes.citation_builder.build_citations",
        lambda ns: [{"rank": 1, "score": 0.9, "text": "t", "vector_node_id": "n1"}],
    )
    monkeypatch.setattr(
        "app.agents.nodes.synthesizer.GroundedAnswerSynthesizer.synthesize",
        lambda self, **kw: {"answer": "Answer after loop.", "reasoning": None, "images_used": [], "reasoning_effort_applied": False},
    )

    from app.agents.graph import build_graph

    graph = build_graph()
    final = graph.invoke(make_state())

    assert call_count["n"] == 2  # two retrieval passes
    assert final["answer"] == "Answer after loop."
    assert final["iteration_count"] == 2


def test_graph_forces_synthesis_at_max_iterations(monkeypatch):
    """Graph must stop looping and synthesize when max iterations is reached."""
    monkeypatch.setattr("app.agents.nodes.evidence_evaluator.settings.AGENTIC_MAX_ITERATIONS", 2)
    monkeypatch.setattr(
        "app.agents.nodes.intent_router.invoke_llm_chat",
        lambda **kw: _llm_response("internal"),
    )
    monkeypatch.setattr(
        "app.agents.nodes.vector_retrieval.VecteraRetriever.retrieve_only",
        lambda self, q: [make_node()],
    )
    # Always returns insufficient — max iterations guard must override
    monkeypatch.setattr(
        "app.agents.nodes.evidence_evaluator.invoke_llm_chat",
        lambda **kw: _llm_eval(False, "still not enough"),
    )
    monkeypatch.setattr("app.agents.nodes.reranker.rerank_nodes", lambda ns, **kw: ns)
    monkeypatch.setattr(
        "app.agents.nodes.reranker.VecteraRetriever._select_evidence_nodes",
        lambda self, q, ns: ns,
    )
    monkeypatch.setattr("app.agents.nodes.conflict_detector.detect_conflicts", lambda **kw: [])
    monkeypatch.setattr("app.agents.nodes.citation_builder.build_citations", lambda ns: [])
    monkeypatch.setattr(
        "app.agents.nodes.synthesizer.GroundedAnswerSynthesizer.synthesize",
        lambda self, **kw: {"answer": "Forced answer.", "reasoning": None, "images_used": [], "reasoning_effort_applied": False},
    )

    from app.agents.graph import build_graph

    graph = build_graph()
    final = graph.invoke(make_state())

    assert final["iteration_count"] == 2  # stopped at max
    assert final["answer"] == "Forced answer."


def test_feature_flag_routes_to_adapter(monkeypatch):
    """When ENABLE_AGENTIC_RAG=True, execute_query uses AgenticRetrieverAdapter."""
    captured = {}

    class FakeAdapter:
        def __init__(self, **kw):
            captured["init"] = True

        def query(self, question, status_callback=None):
            captured["queried"] = True
            return {
                "answer": "agentic answer",
                "citations": [],
                "conflicts": [],
                "source_count": 0,
                "evidence_count": 0,
                "images_used": [],
                "image_evidence_count": 0,
                "reasoning_effort": "medium",
                "reasoning_effort_applied": False,
                "retrieval_diagnostics": {},
                "intent_labels": [],
                "evidence_by_document": {},
                "evidence_by_version_group": {},
                "evidence_by_entity": {},
                "agentic_iterations": 1,
            }

    monkeypatch.setattr("app.core.config.settings.ENABLE_AGENTIC_RAG", True)

    import importlib
    import app.services.query_service as qs

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    # Patch the lazy import path
    import sys
    fake_module = MagicMock()
    fake_module.AgenticRetrieverAdapter = FakeAdapter
    sys.modules["app.agents.adapter"] = fake_module

    try:
        # Use in-memory SQLite db session
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.db.base import Base

        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        result = qs.execute_query(
            question="What is FFO?",
            client_id="client-1",
            db=db,
        )
        assert captured.get("queried") is True
        assert result["answer"] == "agentic answer"
    finally:
        del sys.modules["app.agents.adapter"]
        Base.metadata.drop_all(bind=engine)
        db.close()
