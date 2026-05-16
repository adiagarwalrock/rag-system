from __future__ import annotations

import logging
from typing import Any

from app.indexing.vector_store import vector_store_manager
from app.services.evidence_selector import _is_conflict_focused_query
from app.services.query_rewriter import build_query_variants, should_expand_query
from app.components.retriever.fusion import fuse_node_batches

logger = logging.getLogger(__name__)


def retrieve(retriever: Any, question: str) -> tuple[list, dict[str, Any]]:
    try:
        nodes, expanded = retrieve_with_mode(retriever, question, hybrid=True)
        mode = "hybrid"
    except Exception:
        logger.exception(
            "Hybrid retrieval failed for client %s; falling back to dense retrieval",
            retriever.client_id,
        )
        nodes, expanded = retrieve_with_mode(retriever, question, hybrid=False)
        mode = "dense_fallback"

    logger.info(
        "Retrieved %d nodes with mode=%s expansion=%s for client %s",
        len(nodes),
        mode,
        expanded,
        retriever.client_id,
    )
    return nodes, {"retrieval_mode": mode, "query_expanded": expanded}


def retrieve_with_mode(retriever: Any, question: str, hybrid: bool) -> tuple[list, bool]:
    prefetch_top_k = (
        retriever.prefetch_top_k + 10
        if _is_conflict_focused_query(question)
        else retriever.prefetch_top_k
    )
    base_retriever = vector_store_manager.get_retriever(
        filters=retriever.filters,
        similarity_top_k=prefetch_top_k,
        sparse_top_k=prefetch_top_k,
        hybrid_top_k=prefetch_top_k,
        hybrid=hybrid,
    )

    expansion_question = build_query_expansion_input(retriever, question)
    if should_expand_query(expansion_question):
        return retrieve_with_expansion(
            retriever,
            question,
            expansion_question,
            base_retriever,
        )

    return base_retriever.retrieve(question), False


def retrieve_with_expansion(
    retriever: Any,
    question: str,
    expansion_question: dict[str, Any],
    base_retriever: Any,
) -> tuple[list, bool]:
    _ = retriever
    query_variants = build_query_variants(expansion_question)
    if len(query_variants) == 1:
        return base_retriever.retrieve(question), False

    batches = [base_retriever.retrieve(variant) for variant in query_variants]
    return fuse_node_batches(batches), True


def build_query_expansion_input(retriever: Any, question: str) -> dict[str, Any]:
    return {
        "current_question": question,
        "recent_turns": retriever.conversation_context.get("recent_turns") or [],
    }
