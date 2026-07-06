"""
Retriever: orchestrates the full retrieval pipeline.

Flow: vector retrieval -> ranking -> evidence selection -> conflict checks -> citation building

Evidence selection, prompt building, and answer synthesis are delegated
to dedicated modules under ``app.retrieval``.
"""

import logging
from typing import Any, Dict, List

from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

# Backward-compatible re-exports: tests monkeypatch these on retriever_module
from app.agents.agent_base import normalize_reasoning_effort
from app.indexing.vector_store import vector_store_manager
from app.services.citation_builder import build_citations
from app.services.conflict_detector import detect_conflicts

# Sub-module imports — used by RAGRetriever and its callers
from app.services.evidence_selector import (
    _build_retrieval_diagnostics,
    _ensure_image_evidence,
    _ensure_structured_evidence,
    _evidence_diversity_key,
    _has_numeric_signal,
    _is_comparison_or_conflict_query,
    _is_conflict_focused_query,
    _is_cross_document_synthesis_query,
    _is_reasoning_chunk,
    _is_reasoning_priority_query,
    _is_time_anchored_query,
    _node_unique_key,
)
from app.services.query_rewriter import build_query_variants, should_expand_query
from app.services.synthesizer import GroundedAnswerSynthesizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EVIDENCE_LIMIT = 7
COMPARATIVE_EVIDENCE_LIMIT = 10
CONFLICT_EVIDENCE_LIMIT = 8
CROSS_DOCUMENT_EVIDENCE_LIMIT = 14

# Max chunks from the same (document_id, page_num) allowed in evidence.
# Prevents a single semantically-dominant page from flooding all evidence slots.
MAX_CHUNKS_PER_PAGE = 2


def _page_diversity_key(node: Any) -> str:
    """Return a key that groups chunks by their source document + page."""
    metadata = node.node.metadata or {}
    doc_id = metadata.get("document_id") or metadata.get("source_file") or "unknown_doc"
    page = metadata.get("page_num") or metadata.get("slide_num") or "unknown_page"
    return f"{doc_id}:{page}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class RAGRetriever:
    """
    Full retrieval pipeline with client-scoped filtering,
    reranking, temporal awareness, conflict detection, and citations.
    """

    def __init__(
        self,
        client_id: str,
        top_k: int = 15,
        reasoning_effort: str = "medium",
        conversation_context: dict[str, Any] | None = None,
    ):
        self.client_id = client_id
        self.top_k = top_k
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self.conversation_context = conversation_context or {}
        self.prefetch_top_k = top_k * 5
        self.evidence_limit = DEFAULT_EVIDENCE_LIMIT
        self.comparative_evidence_limit = COMPARATIVE_EVIDENCE_LIMIT
        self.conflict_evidence_limit = CONFLICT_EVIDENCE_LIMIT
        self.filters = MetadataFilters(
            filters=[ExactMatchFilter(key="client_id", value=self.client_id)]
        )
        self.answer_synthesizer = GroundedAnswerSynthesizer(
            client_id=self.client_id,
            reasoning_effort=self.reasoning_effort,
            conversation_context=self.conversation_context,
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute the full retrieval and answer pipeline.

        Returns:
            Dict with answer, citations, conflicts, and metadata.
        """
        logger.info("Query for client %s: %s", self.client_id, question[:100])

        # Step 1: Hybrid Qdrant retrieval, with dense fallback for older indexes.
        source_nodes, retrieval_metadata = self._retrieve(question)

        if not source_nodes:
            return {
                "answer": "I could not find relevant information in the uploaded documents to answer this question.",
                "reasoning": "",
                "citations": [],
                "conflicts": [],
                "source_count": 0,
                "images_used": [],
                "image_evidence_count": 0,
                "reasoning_effort": self.reasoning_effort,
                "reasoning_effort_applied": False,
                "retrieval_diagnostics": _build_retrieval_diagnostics([], []),
                **retrieval_metadata,
            }

        # Step 2: Rank candidates with semantic + temporal/version signals.
        ranked_nodes = self._rank_nodes(question, source_nodes)

        # Step 3: Select bounded evidence used for answer synthesis/citations.
        evidence_nodes = self._select_evidence_nodes(question, ranked_nodes)
        retrieval_diagnostics = _build_retrieval_diagnostics(
            ranked_nodes, evidence_nodes
        )

        # Step 4: Conflict detection (advisory, scoped to evidence + near-miss nodes).
        conflicts = detect_conflicts(
            ranked_nodes, evidence_nodes=evidence_nodes, question=question
        )

        # Step 5: Build citations only from the selected answer evidence.
        citations = build_citations(evidence_nodes)

        # Step 6: Synthesize grounded answer from selected evidence only.
        synthesis = self._synthesize_answer(question, citations, conflicts)

        return {
            "answer": synthesis["answer"],
            "reasoning": synthesis.get("reasoning"),
            "citations": citations,
            "conflicts": conflicts,
            "source_count": len(ranked_nodes),
            "evidence_count": len(evidence_nodes),
            "images_used": synthesis.get("images_used", []),
            "image_evidence_count": len(synthesis.get("images_used", [])),
            "reasoning_effort": self.reasoning_effort,
            "reasoning_effort_applied": synthesis.get(
                "reasoning_effort_applied", False
            ),
            "retrieval_diagnostics": retrieval_diagnostics,
            **retrieval_metadata,
        }

    def retrieve_only(self, question: str, rerank: bool = True) -> List:
        """Retrieve source nodes without generating an answer."""
        source_nodes, _ = self._retrieve(question)
        if rerank:
            return self._rank_nodes(question, source_nodes)
        return source_nodes

    def _rank_nodes(self, question: str, source_nodes: list) -> list:
        from app.components.reranker import rerank_nodes

        prefer_latest = not (
            _is_comparison_or_conflict_query(question)
            or _is_time_anchored_query(question)
        )
        rank_top_k = (
            self.top_k + 8 if _is_conflict_focused_query(question) else self.top_k
        )
        return rerank_nodes(
            source_nodes,
            top_k=rank_top_k,
            prefer_latest=prefer_latest,
            query=question,
        )

    def _retrieve(self, question: str) -> tuple[list, dict[str, Any]]:
        try:
            nodes, expanded = self._retrieve_with_mode(question, hybrid=True)
            mode = "hybrid"
        except Exception:
            logger.exception(
                "Hybrid retrieval failed for client %s; falling back to dense retrieval",
                self.client_id,
            )
            nodes, expanded = self._retrieve_with_mode(question, hybrid=False)
            mode = "dense_fallback"

        logger.info(
            "Retrieved %d nodes with mode=%s expansion=%s for client %s",
            len(nodes),
            mode,
            expanded,
            self.client_id,
        )
        return nodes, {"retrieval_mode": mode, "query_expanded": expanded}

    def _retrieve_with_mode(self, question: str, hybrid: bool) -> tuple[list, bool]:
        prefetch_top_k = self.prefetch_top_k
        if _is_conflict_focused_query(question):
            prefetch_top_k += 10
        base_retriever = vector_store_manager.get_retriever(
            filters=self.filters,
            similarity_top_k=prefetch_top_k,
            sparse_top_k=prefetch_top_k,
            hybrid_top_k=prefetch_top_k,
            hybrid=hybrid,
        )

        expansion_question = self._build_query_expansion_input(question)
        should_expand = should_expand_query(expansion_question)
        if should_expand:
            return self._retrieve_with_expansion(
                question,
                expansion_question,
                base_retriever,
            )

        return base_retriever.retrieve(question), False

    def _retrieve_with_expansion(
        self,
        question: str,
        expansion_question: dict[str, Any],
        base_retriever,
    ) -> tuple[list, bool]:
        query_variants = build_query_variants(expansion_question)
        if len(query_variants) == 1:
            return base_retriever.retrieve(question), False

        batches = [base_retriever.retrieve(variant) for variant in query_variants]
        return _fuse_node_batches(batches), True

    def _build_query_expansion_input(self, question: str) -> dict[str, Any]:
        return {
            "current_question": question,
            "recent_turns": self.conversation_context.get("recent_turns") or [],
        }

    def _select_evidence_nodes(self, question: str, ranked_nodes: list) -> list:
        if not ranked_nodes:
            return []

        comparative_query = _is_comparison_or_conflict_query(question)
        conflict_focused_query = _is_conflict_focused_query(question)
        cross_document_query = _is_cross_document_synthesis_query(ranked_nodes)
        evidence_cap = self._resolve_evidence_cap(
            comparative_query=comparative_query,
            conflict_focused_query=conflict_focused_query,
            cross_document_query=cross_document_query,
        )
        candidate_nodes = self._prioritize_conflict_candidates(
            ranked_nodes=ranked_nodes,
            conflict_focused_query=conflict_focused_query,
        )
        primary_candidates, secondary_candidates = self._partition_reasoning_candidates(
            question=question,
            candidate_nodes=candidate_nodes,
        )
        selected = self._collect_evidence_candidates(
            primary_candidates=primary_candidates,
            secondary_candidates=secondary_candidates,
            evidence_cap=evidence_cap,
            comparative_query=comparative_query,
        )
        selected = _ensure_structured_evidence(
            question=question,
            selected_nodes=selected,
            ranked_nodes=ranked_nodes,
            evidence_cap=evidence_cap,
        )
        selected = _ensure_image_evidence(
            question=question,
            selected_nodes=selected,
            ranked_nodes=ranked_nodes,
            evidence_cap=evidence_cap,
        )
        return selected

    def _resolve_evidence_cap(
        self,
        *,
        comparative_query: bool,
        conflict_focused_query: bool,
        cross_document_query: bool = False,
    ) -> int:
        if cross_document_query:
            return CROSS_DOCUMENT_EVIDENCE_LIMIT
        if conflict_focused_query:
            return self.conflict_evidence_limit
        if comparative_query:
            return self.comparative_evidence_limit
        return self.evidence_limit

    def _prioritize_conflict_candidates(
        self,
        *,
        ranked_nodes: list,
        conflict_focused_query: bool,
    ) -> list:
        if not conflict_focused_query:
            return ranked_nodes

        # For conflict queries, prioritize chunks that actually carry numeric signals.
        numeric_nodes = [node for node in ranked_nodes if _has_numeric_signal(node)]
        non_numeric_nodes = [
            node for node in ranked_nodes if not _has_numeric_signal(node)
        ]
        return numeric_nodes + non_numeric_nodes

    def _partition_reasoning_candidates(
        self,
        *,
        question: str,
        candidate_nodes: list,
    ) -> tuple[list, list]:
        if _is_reasoning_priority_query(question):
            return candidate_nodes, []

        # For comparison/conflict queries, keep factual chunks leading first.
        if _is_comparison_or_conflict_query(question):
            primary_candidates = [
                node for node in candidate_nodes if not _is_reasoning_chunk(node)
            ]
            secondary_candidates = [
                node for node in candidate_nodes if _is_reasoning_chunk(node)
            ]
            return primary_candidates, secondary_candidates

        # For all other queries, promote high-scoring reasoning chunks to primary.
        cutoff = max(1, len(candidate_nodes) // 2)
        primary_candidates = []
        secondary_candidates = []
        for rank, node in enumerate(candidate_nodes):
            if _is_reasoning_chunk(node) and rank >= cutoff:
                secondary_candidates.append(node)
            else:
                primary_candidates.append(node)
        return primary_candidates, secondary_candidates

    def _collect_evidence_candidates(
        self,
        *,
        primary_candidates: list,
        secondary_candidates: list,
        evidence_cap: int,
        comparative_query: bool,
    ) -> list:
        selected: list[Any] = []
        selected_keys: set[str] = set()
        candidate_batches = [primary_candidates]
        if secondary_candidates:
            candidate_batches.append(secondary_candidates)

        if comparative_query:
            selected_diversity_keys: set[str] = set()
            if self._collect_diverse_nodes(
                candidate_batches=candidate_batches,
                selected=selected,
                selected_keys=selected_keys,
                selected_diversity_keys=selected_diversity_keys,
                evidence_cap=evidence_cap,
            ):
                return selected

        self._collect_unique_nodes(
            candidate_batches=candidate_batches,
            selected=selected,
            selected_keys=selected_keys,
            evidence_cap=evidence_cap,
        )
        return selected

    def _collect_diverse_nodes(
        self,
        *,
        candidate_batches: list[list[Any]],
        selected: list[Any],
        selected_keys: set[str],
        selected_diversity_keys: set[str],
        evidence_cap: int,
    ) -> bool:
        # Prefer source diversity first: version label when available, otherwise
        # document family/document ID as fallback.
        for batch in candidate_batches:
            for node in batch:
                diversity_key = _evidence_diversity_key(node)
                if diversity_key in selected_diversity_keys:
                    continue
                key = _node_unique_key(node)
                if key in selected_keys:
                    continue
                selected.append(node)
                selected_keys.add(key)
                selected_diversity_keys.add(diversity_key)
                if len(selected) >= evidence_cap:
                    return True
        return False

    def _collect_unique_nodes(
        self,
        *,
        candidate_batches: list[list[Any]],
        selected: list[Any],
        selected_keys: set[str],
        evidence_cap: int,
        max_per_page: int = MAX_CHUNKS_PER_PAGE,
    ) -> None:
        # Track how many chunks are already selected per (document_id, page_num) pair
        page_counts: dict[str, int] = {}
        for node in selected:
            page_key = _page_diversity_key(node)
            page_counts[page_key] = page_counts.get(page_key, 0) + 1

        # First pass: enforce per-page cap for diversity
        for batch in candidate_batches:
            for node in batch:
                if len(selected) >= evidence_cap:
                    return
                key = _node_unique_key(node)
                if key in selected_keys:
                    continue
                page_key = _page_diversity_key(node)
                if page_counts.get(page_key, 0) >= max_per_page:
                    continue
                selected.append(node)
                selected_keys.add(key)
                page_counts[page_key] = page_counts.get(page_key, 0) + 1

        # Second pass without cap: fill remaining slots if diversity pass left gaps
        if len(selected) < evidence_cap:
            for batch in candidate_batches:
                for node in batch:
                    if len(selected) >= evidence_cap:
                        return
                    key = _node_unique_key(node)
                    if key in selected_keys:
                        continue
                    selected.append(node)
                    selected_keys.add(key)

    def _synthesize_answer(
        self,
        question: str,
        citations: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.answer_synthesizer.synthesize(
            question=question,
            citations=citations,
            conflicts=conflicts,
        )


# ---------------------------------------------------------------------------
# Node fusion (used by _retrieve_with_expansion)
# ---------------------------------------------------------------------------


def _fuse_node_batches(node_batches: list[list]) -> list:
    fused: dict[str, tuple[float, Any]] = {}

    for batch in node_batches:
        for rank, node in enumerate(batch, start=1):
            key = _node_fusion_key(node, rank)
            retrieval_score, fused_score = _initial_fused_score(node, rank)

            existing = fused.get(key)
            if existing:
                fused_score += existing[0]
                node = _choose_higher_scored_node(existing[1], node, retrieval_score)

            node.score = fused_score
            fused[key] = (fused_score, node)

    return [
        node
        for _, node in sorted(fused.values(), key=lambda item: item[0], reverse=True)
    ]


def _node_fusion_key(node: Any, rank: int) -> str:
    node_id = getattr(node.node, "node_id", None)
    metadata = node.node.metadata or {}
    key = node_id or metadata.get("chunk_id") or metadata.get("document_id")
    if key:
        return str(key)
    return f"{metadata.get('file_name', 'unknown')}:{rank}:{node.node.text[:64]}"


def _initial_fused_score(node: Any, rank: int) -> tuple[float, float]:
    retrieval_score = float(node.score or 0.0)
    rrf_score = 1.0 / (60 + rank)
    return retrieval_score, retrieval_score + rrf_score


def _choose_higher_scored_node(
    existing_node: Any, candidate_node: Any, candidate_retrieval_score: float
) -> Any:
    return (
        existing_node
        if (existing_node.score or 0.0) >= candidate_retrieval_score
        else candidate_node
    )
