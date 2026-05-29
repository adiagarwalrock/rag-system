"""
LangGraph StateGraph for the agentic RAG pipeline.

Graph topology:
    START → intent_router → vector_retrieval → evidence_evaluator
                                    ↑                   |
                                    └── (loop back) ────┘ (not sufficient)
                                                        |
                                              (sufficient or max iter)
                                                        ↓
                                    reranker → conflict_detector → citation_builder → synthesizer → END
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    citation_builder_node,
    conflict_detector_node,
    evidence_evaluator_node,
    intent_router_node,
    reranker_node,
    synthesizer_node,
    vector_retrieval_node,
)
from app.agents.state import AgentState
from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Conditional edge functions ────────────────────────────────────────────────


def _route_after_evaluation(state: AgentState) -> str:
    if state.get("evidence_sufficient"):
        return "reranker"
    if (state.get("iteration_count") or 0) >= settings.AGENTIC_MAX_ITERATIONS:
        return "reranker"
    return "vector_retrieval"


# ── Graph construction ────────────────────────────────────────────────────────


def build_graph() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("intent_router", intent_router_node)
    graph.add_node("vector_retrieval", vector_retrieval_node)
    graph.add_node("evidence_evaluator", evidence_evaluator_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("conflict_detector", conflict_detector_node)
    graph.add_node("citation_builder", citation_builder_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "intent_router")
    graph.add_edge("intent_router", "vector_retrieval")
    graph.add_edge("vector_retrieval", "evidence_evaluator")

    graph.add_conditional_edges(
        "evidence_evaluator",
        _route_after_evaluation,
        {
            "reranker": "reranker",
            "vector_retrieval": "vector_retrieval",
        },
    )

    graph.add_edge("reranker", "conflict_detector")
    graph.add_edge("conflict_detector", "citation_builder")
    graph.add_edge("citation_builder", "synthesizer")
    graph.add_edge("synthesizer", END)

    compiled = graph.compile()
    mermaid = compiled.get_graph().draw_mermaid()
    logger.info("Agentic RAG graph (Mermaid):\n%s", mermaid)
    with open("agent_graph.mmd", "w") as f:
        f.write(mermaid)
    return compiled


@lru_cache(maxsize=1)
def get_compiled_graph() -> Any:
    """Compiled graph singleton — lazy on first agentic request, cached thereafter."""
    logger.info("Compiling agentic RAG LangGraph (one-time)")
    return build_graph()
