"""
AgenticRetrieverAdapter: duck-types VecteraRetriever so that QueryExecutionService
can use the LangGraph agent pipeline as a drop-in retriever_factory.

QueryExecutionService calls:
    retriever = retriever_factory(client_id=..., reasoning_effort=..., ...)
    result = retriever.query(question, status_callback=...)

This adapter builds the initial AgentState, invokes the compiled graph, and
returns a dict with the same shape as VecteraRetriever.query().
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from app.agents.graph import get_compiled_graph
from app.agents.state import AgentState

logger = logging.getLogger(__name__)


class AgenticRetrieverAdapter:
    """Drop-in replacement for VecteraRetriever, backed by the LangGraph agent graph."""

    def __init__(
        self,
        client_id: str,
        reasoning_effort: str = "medium",
        reasoning_summary: str | None = None,
        conversation_context: dict[str, Any] | None = None,
        reasoning_callback: Callable[[str], None] | None = None,
        **_kwargs: Any,
    ):
        self._client_id = client_id
        self._reasoning_effort = reasoning_effort
        self._reasoning_summary = reasoning_summary
        self._conversation_context = conversation_context or {}
        self._reasoning_callback = reasoning_callback

    def query(
        self,
        question: str,
        status_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        initial_state: AgentState = {
            # Inputs
            "question": question,
            "client_id": self._client_id,
            "reasoning_effort": self._reasoning_effort,
            "reasoning_summary": self._reasoning_summary,
            "conversation_context": self._conversation_context,
            "status_callback": status_callback,
            "reasoning_callback": self._reasoning_callback,
            # Intent (populated by intent_router_node)
            "intent_labels": [],
            "route": "internal",
            # Retrieval accumulation (start empty; operator.add appends)
            "retrieved_nodes": [],
            "retrieved_node_ids": [],
            # Post-loop (populated after loop exits)
            "evidence_nodes": [],
            "citations": [],
            "conflicts": [],
            # Answer
            "answer": "",
            "reasoning": None,
            "images_used": [],
            "reasoning_effort_applied": False,
            # Loop control
            "iteration_count": 0,
            "evidence_sufficient": False,
            "retrieval_gap": None,
        }

        graph = get_compiled_graph()
        final_state: AgentState = graph.invoke(initial_state)

        return _build_result(final_state, self._reasoning_effort)


def _build_result(state: AgentState, reasoning_effort: str) -> dict[str, Any]:
    """Convert final AgentState into the same dict shape as VecteraRetriever.query()."""
    evidence_nodes = state.get("evidence_nodes") or []
    retrieved_nodes = state.get("retrieved_nodes") or []
    images_used = state.get("images_used") or []

    return {
        "answer": state.get("answer") or "I could not generate an answer.",
        "reasoning": state.get("reasoning"),
        "citations": state.get("citations") or [],
        "conflicts": state.get("conflicts") or [],
        "source_count": len(retrieved_nodes),
        "evidence_count": len(evidence_nodes),
        "images_used": images_used,
        "image_evidence_count": len(images_used),
        "reasoning_effort": reasoning_effort,
        "reasoning_effort_applied": state.get("reasoning_effort_applied", False),
        "retrieval_diagnostics": {},
        "intent_labels": state.get("intent_labels") or [],
        "evidence_by_document": {},
        "evidence_by_version_group": {},
        "evidence_by_entity": {},
        # Agentic-specific observability keys (ignored by existing consumers)
        "agentic_iterations": state.get("iteration_count", 0),
    }
