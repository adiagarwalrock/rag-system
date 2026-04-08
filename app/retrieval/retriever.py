"""
Retriever: orchestrates the full retrieval pipeline.

Flow: vector retrieval -> ranking -> evidence selection -> conflict checks -> citation building
"""

import logging
from typing import Any, Dict, List

from llama_index.core import Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from app.indexing.vector_store import get_retriever, is_placeholder_mode
from app.retrieval.citation_builder import build_citations
from app.retrieval.conflict_detector import detect_conflicts
from app.retrieval.query_expansion import build_query_variants, should_expand_query
from app.retrieval.reranker import rerank_nodes

logger = logging.getLogger(__name__)

DEFAULT_EVIDENCE_LIMIT = 4
COMPARATIVE_EVIDENCE_LIMIT = 6
CONFLICT_EVIDENCE_LIMIT = 8
COMPARATIVE_TERMS = (
    "compare",
    "comparison",
    "difference",
    "changes",
    "changed",
    "conflict",
    "conflicting",
    "contradict",
    "versus",
    "vs",
    "between",
)


class VecteraRetriever:
    """
    Full retrieval pipeline with client-scoped filtering,
    reranking, temporal awareness, conflict detection, and citations.
    """

    def __init__(self, client_id: str, top_k: int = 10):
        self.client_id = client_id
        self.top_k = top_k
        self.prefetch_top_k = top_k + 5
        self.evidence_limit = DEFAULT_EVIDENCE_LIMIT
        self.comparative_evidence_limit = COMPARATIVE_EVIDENCE_LIMIT
        self.conflict_evidence_limit = CONFLICT_EVIDENCE_LIMIT
        self.filters = MetadataFilters(
            filters=[ExactMatchFilter(key="client_id", value=self.client_id)]
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
                "citations": [],
                "conflicts": [],
                "source_count": 0,
            }

        # Step 2: Rank candidates with semantic + temporal/version signals.
        ranked_nodes = self._rank_nodes(question, source_nodes)

        # Step 3: Select bounded evidence used for answer synthesis/citations.
        evidence_nodes = self._select_evidence_nodes(question, ranked_nodes)

        # Step 4: Conflict detection (advisory, scoped to evidence + near-miss nodes).
        conflicts = detect_conflicts(
            ranked_nodes, evidence_nodes=evidence_nodes, question=question
        )

        # Step 5: Build citations only from the selected answer evidence.
        citations = build_citations(evidence_nodes)

        # Step 6: Synthesize grounded answer from selected evidence only.
        answer_text = self._synthesize_answer(question, citations, conflicts)

        return {
            "answer": answer_text,
            "citations": citations,
            "conflicts": conflicts,
            "source_count": len(ranked_nodes),
            "evidence_count": len(evidence_nodes),
            **retrieval_metadata,
        }

    def retrieve_only(self, question: str) -> List:
        """Retrieve source nodes without generating an answer."""
        source_nodes, _ = self._retrieve(question)
        return self._rank_nodes(question, source_nodes)

    def _rank_nodes(self, question: str, source_nodes: list) -> list:
        prefer_latest = not _is_comparison_or_conflict_query(question)
        rank_top_k = (
            self.top_k + 8 if _is_conflict_focused_query(question) else self.top_k
        )
        return rerank_nodes(source_nodes, top_k=rank_top_k, prefer_latest=prefer_latest)

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
        prefetch_top_k = (
            self.prefetch_top_k + 10
            if _is_conflict_focused_query(question)
            else self.prefetch_top_k
        )
        base_retriever = get_retriever(
            filters=self.filters,
            similarity_top_k=prefetch_top_k,
            sparse_top_k=prefetch_top_k,
            hybrid_top_k=prefetch_top_k,
            hybrid=hybrid,
        )

        should_expand = should_expand_query(question) and not is_placeholder_mode()
        if should_expand:
            return self._retrieve_with_expansion(
                question, base_retriever, prefetch_top_k
            )

        return base_retriever.retrieve(question), False

    def _retrieve_with_expansion(
        self, question: str, base_retriever, prefetch_top_k: int
    ) -> tuple[list, bool]:
        try:
            fusion_retriever = QueryFusionRetriever(
                [base_retriever],
                similarity_top_k=prefetch_top_k,
                num_queries=3,
                mode=FUSION_MODES.RECIPROCAL_RANK,
                use_async=False,
                verbose=False,
            )
            return fusion_retriever.retrieve(question), True
        except Exception:
            logger.exception(
                "LlamaIndex query fusion expansion failed; using manual expansion fallback"
            )
            query_variants = build_query_variants(question)
            batches = [base_retriever.retrieve(variant) for variant in query_variants]
            return _fuse_node_batches(batches), len(query_variants) > 1

    def _select_evidence_nodes(self, question: str, ranked_nodes: list) -> list:
        if not ranked_nodes:
            return []

        comparative_query = _is_comparison_or_conflict_query(question)
        conflict_focused_query = _is_conflict_focused_query(question)
        evidence_cap = self.evidence_limit
        if comparative_query:
            evidence_cap = self.comparative_evidence_limit
        if conflict_focused_query:
            evidence_cap = self.conflict_evidence_limit

        candidate_nodes = ranked_nodes
        if conflict_focused_query:
            # For conflict queries, prioritize chunks that actually carry numeric signals.
            numeric_nodes = [node for node in ranked_nodes if _has_numeric_signal(node)]
            non_numeric_nodes = [
                node for node in ranked_nodes if not _has_numeric_signal(node)
            ]
            candidate_nodes = numeric_nodes + non_numeric_nodes

        selected = []
        selected_keys: set[str] = set()
        selected_diversity_keys: set[str] = set()

        if comparative_query:
            # Prefer source diversity first: version label when available, otherwise
            # document family/document ID as fallback.
            for node in candidate_nodes:
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
                    return selected

        for node in candidate_nodes:
            if len(selected) >= evidence_cap:
                break
            key = _node_unique_key(node)
            if key in selected_keys:
                continue
            selected.append(node)
            selected_keys.add(key)

        return selected

    def _synthesize_answer(
        self,
        question: str,
        citations: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
    ) -> str:
        if not citations:
            return (
                "I could not find enough relevant evidence in the uploaded documents "
                "to answer this question confidently."
            )

        try:
            prompt = _build_grounded_prompt(question, citations, conflicts)
            response = Settings.llm.complete(prompt)
            answer = str(response).strip()
            if answer:
                return answer
            raise ValueError("LLM returned empty answer")
        except Exception:
            logger.exception(
                "Answer synthesis failed; returning source-grounded fallback"
            )
            labels = []
            for citation in citations[:3]:
                label = citation.get("citation_label") or citation.get("document_name")
                if label:
                    labels.append(label)
            if labels:
                return (
                    "I found relevant evidence, but could not synthesize a final answer. "
                    f"Review these sources: {', '.join(labels)}."
                )
            return "I found relevant evidence, but could not synthesize a final answer."


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


def _is_comparison_or_conflict_query(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in COMPARATIVE_TERMS)


def _is_conflict_focused_query(question: str) -> bool:
    normalized = question.lower()
    conflict_terms = (
        "conflict",
        "conflicting",
        "contradict",
        "contradiction",
        "across documents",
        "across sources",
        "disagree",
    )
    return any(term in normalized for term in conflict_terms)


def _has_numeric_signal(node: Any) -> bool:
    metadata = node.node.metadata or {}
    raw = metadata.get("contains_numeric_data")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "1", "yes"}
    return False


def _evidence_diversity_key(node: Any) -> str:
    metadata = node.node.metadata or {}
    version_label = (metadata.get("version_label") or "").strip().lower()
    if version_label:
        return f"version:{version_label}"

    version_group = metadata.get("document_version_group") or metadata.get(
        "document_family"
    )
    if version_group:
        return f"group:{str(version_group).strip().lower()}"

    document_id = metadata.get("document_id")
    if document_id:
        return f"doc:{document_id}"

    doc_name = metadata.get("document_name") or metadata.get("source_file")
    if doc_name:
        return f"name:{str(doc_name).strip().lower()}"

    return f"node:{_node_unique_key(node)}"


def _node_unique_key(node: Any) -> str:
    metadata = node.node.metadata or {}
    return str(
        getattr(node.node, "node_id", None)
        or metadata.get("chunk_id")
        or metadata.get("document_id")
        or metadata.get("citation_label")
        or node.node.text[:80]
    )


def _build_grounded_prompt(
    question: str, citations: list[dict[str, Any]], conflicts: list[dict[str, Any]]
) -> str:
    evidence_lines = []
    for index, citation in enumerate(citations, start=1):
        label = citation.get("citation_label") or citation.get(
            "document_name", f"Source {index}"
        )
        version = citation.get("version_label") or "unknown"
        excerpt = (citation.get("text") or "").strip().replace("\n", " ")
        evidence_lines.append(
            f"[{index}] {label} | version={version} | excerpt={excerpt[:450]}"
        )

    conflict_lines = []
    for conflict in conflicts[:2]:
        summary = (conflict.get("summary") or "").strip()
        if summary:
            conflict_lines.append(f"- {summary}")

    conflict_block = (
        "\n".join(conflict_lines)
        if conflict_lines
        else (
            "- No high-confidence conflicts were detected in the selected excerpts. "
            "This is not proof that all documents are conflict-free."
        )
    )
    evidence_block = "\n".join(evidence_lines)

    return (
        "You are a retrieval-grounded assistant for sensitive enterprise documents.\n"
        "Answer using only the provided evidence snippets.\n"
        "Rules:\n"
        "1) If evidence is insufficient or contradictory, say so explicitly.\n"
        "2) For each factual claim, cite at least one source index like [1].\n"
        "3) Do not cite sources that are not in the evidence list.\n"
        "4) Prefer the most current/effective version unless the question asks for comparison.\n"
        "5) If conflict hints are empty, avoid absolute claims such as "
        "'no conflicts exist'; state only what was or was not detected in "
        "the retrieved evidence.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Conflict hints:\n{conflict_block}\n\n"
        "Answer:"
    )
