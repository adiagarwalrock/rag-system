"""
vector_retrieval_node: retrieves fresh nodes from Qdrant, excluding already-seen IDs.

retrieve_only() is called unchanged. Exclusion is handled in Python after retrieval
so no Qdrant schema changes are needed. iteration_count is incremented here, after
retrieval, so AGENTIC_MAX_ITERATIONS=5 means exactly 5 retrieval passes.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.nodes._shared import emit_status
from app.retrieval.retriever import VecteraRetriever

logger = logging.getLogger(__name__)


def vector_retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
    iteration = state.get("iteration_count", 0)
    excluded_ids: set[str] = set(state.get("retrieved_node_ids") or [])

    # Query priority: planned sub-query for this pass > gap reframe > original question
    planned = state.get("planned_queries") or []
    if planned and iteration < len(planned):
        query = planned[iteration]
    elif state.get("retrieval_gap"):
        query = state["retrieval_gap"]
    else:
        query = state["question"]

    if query != state["question"]:
        logger.info("Agentic retrieval iteration %d using query: %s", iteration + 1, query[:80])

    emit_status(state, f"Retrieving relevant chunks (pass {iteration + 1})…")

    retriever = VecteraRetriever(
        client_id=state["client_id"],
        reasoning_effort=state.get("reasoning_effort", "medium"),
        reasoning_summary=state.get("reasoning_summary"),
        conversation_context=state.get("conversation_context") or {},
        reasoning_callback=state.get("reasoning_callback"),
        answer_callback=state.get("answer_callback"),
    )

    all_nodes = retriever.retrieve_only(query)

    # Python-level exclusion — keeps retrieve_only() completely untouched
    new_nodes = [n for n in all_nodes if n.node.node_id not in excluded_ids]
    new_ids = [n.node.node_id for n in new_nodes]

    logger.info(
        "Agentic retrieval pass %d: fetched=%d excluded=%d fresh=%d",
        iteration + 1,
        len(all_nodes),
        len(all_nodes) - len(new_nodes),
        len(new_nodes),
    )

    return {
        "retrieved_nodes": new_nodes,
        "retrieved_node_ids": new_ids,
        "iteration_count": iteration + 1,
    }


