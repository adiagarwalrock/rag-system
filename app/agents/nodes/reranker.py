"""
reranker_node: runs once after the retrieval loop exits on the full accumulated
retrieved_nodes. Wraps rerank_nodes() and VecteraRetriever._select_evidence_nodes().

VecteraRetriever is instantiated minimally (no network/LLM calls in __init__)
solely to access _select_evidence_nodes().

TODO: refactor _select_evidence_nodes into a classmethod on VecteraRetriever
      so this adapter instantiation is no longer needed.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.nodes._shared import emit_status
from app.retrieval.reranker import rerank_nodes
from app.retrieval.retriever import VecteraRetriever

logger = logging.getLogger(__name__)


def reranker_node(state: dict[str, Any]) -> dict[str, Any]:
    retrieved_nodes = state.get("retrieved_nodes") or []
    question = state["question"]

    emit_status(state, f"Reranking {len(retrieved_nodes)} accumulated candidates…")

    if not retrieved_nodes:
        logger.warning("Reranker received empty retrieved_nodes; skipping")
        return {"evidence_nodes": []}

    ranked_nodes = rerank_nodes(
        retrieved_nodes,
        top_k=15,
        prefer_latest=True,
        query=question,
    )

    # TODO: refactor _select_evidence_nodes into a classmethod on VecteraRetriever
    retriever = VecteraRetriever(client_id=state["client_id"])
    evidence_nodes = retriever._select_evidence_nodes(question, ranked_nodes)

    logger.info(
        "Reranker: ranked=%d evidence_selected=%d",
        len(ranked_nodes),
        len(evidence_nodes),
    )

    return {"evidence_nodes": evidence_nodes}


