from __future__ import annotations

from typing import Any

from app.services.evidence_selector import (
    _ensure_image_evidence,
    _ensure_structured_evidence,
    _evidence_diversity_key,
    _has_numeric_signal,
    _is_comparison_or_conflict_query,
    _is_conflict_focused_query,
    _is_reasoning_chunk,
    _is_reasoning_priority_query,
)

MAX_CHUNKS_PER_PAGE = 2


def page_diversity_key(node: Any) -> str:
    metadata = node.node.metadata or {}
    doc_id = metadata.get("document_id") or metadata.get("source_file") or "unknown_doc"
    page = metadata.get("page_num") or metadata.get("slide_num") or "unknown_page"
    return f"{doc_id}:{page}"


def select_evidence_nodes(retriever: Any, question: str, ranked_nodes: list) -> list:
    if not ranked_nodes:
        return []

    comparative_query = _is_comparison_or_conflict_query(question)
    conflict_focused_query = _is_conflict_focused_query(question)
    evidence_cap = resolve_evidence_cap(
        retriever,
        comparative_query=comparative_query,
        conflict_focused_query=conflict_focused_query,
    )
    candidate_nodes = prioritize_conflict_candidates(
        ranked_nodes=ranked_nodes,
        conflict_focused_query=conflict_focused_query,
    )
    primary_candidates, secondary_candidates = partition_reasoning_candidates(
        question=question,
        candidate_nodes=candidate_nodes,
    )
    selected = collect_evidence_candidates(
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


def resolve_evidence_cap(
    retriever: Any,
    *,
    comparative_query: bool,
    conflict_focused_query: bool,
) -> int:
    if conflict_focused_query:
        return retriever.conflict_evidence_limit
    if comparative_query:
        return retriever.comparative_evidence_limit
    return retriever.evidence_limit


def prioritize_conflict_candidates(*, ranked_nodes: list, conflict_focused_query: bool) -> list:
    if not conflict_focused_query:
        return ranked_nodes
    numeric_nodes = [node for node in ranked_nodes if _has_numeric_signal(node)]
    non_numeric_nodes = [node for node in ranked_nodes if not _has_numeric_signal(node)]
    return numeric_nodes + non_numeric_nodes


def partition_reasoning_candidates(*, question: str, candidate_nodes: list) -> tuple[list, list]:
    if _is_reasoning_priority_query(question):
        return candidate_nodes, []

    primary_candidates = [
        node for node in candidate_nodes if not _is_reasoning_chunk(node)
    ]
    secondary_candidates = [
        node for node in candidate_nodes if _is_reasoning_chunk(node)
    ]
    return primary_candidates, secondary_candidates


def collect_evidence_candidates(
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
        if collect_diverse_nodes(
            candidate_batches=candidate_batches,
            selected=selected,
            selected_keys=selected_keys,
            selected_diversity_keys=selected_diversity_keys,
            evidence_cap=evidence_cap,
        ):
            return selected

    collect_unique_nodes(
        candidate_batches=candidate_batches,
        selected=selected,
        selected_keys=selected_keys,
        evidence_cap=evidence_cap,
    )
    return selected


def collect_diverse_nodes(
    *,
    candidate_batches: list[list[Any]],
    selected: list[Any],
    selected_keys: set[str],
    selected_diversity_keys: set[str],
    evidence_cap: int,
) -> bool:
    for batch in candidate_batches:
        for node in batch:
            diversity_key = _evidence_diversity_key(node)
            if diversity_key in selected_diversity_keys:
                continue
            key = node_unique_key(node)
            if key in selected_keys:
                continue
            selected.append(node)
            selected_keys.add(key)
            selected_diversity_keys.add(diversity_key)
            if len(selected) >= evidence_cap:
                return True
    return False


def collect_unique_nodes(
    *,
    candidate_batches: list[list[Any]],
    selected: list[Any],
    selected_keys: set[str],
    evidence_cap: int,
    max_per_page: int = MAX_CHUNKS_PER_PAGE,
) -> None:
    page_counts: dict[str, int] = {}
    for node in selected:
        page_key = page_diversity_key(node)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1

    for batch in candidate_batches:
        for node in batch:
            if len(selected) >= evidence_cap:
                return
            key = node_unique_key(node)
            if key in selected_keys:
                continue
            page_key = page_diversity_key(node)
            if page_counts.get(page_key, 0) >= max_per_page:
                continue
            selected.append(node)
            selected_keys.add(key)
            page_counts[page_key] = page_counts.get(page_key, 0) + 1

    if len(selected) < evidence_cap:
        for batch in candidate_batches:
            for node in batch:
                if len(selected) >= evidence_cap:
                    return
                key = node_unique_key(node)
                if key in selected_keys:
                    continue
                selected.append(node)
                selected_keys.add(key)


def node_unique_key(node: Any) -> str:
    metadata = node.node.metadata or {}
    return str(
        getattr(node.node, "node_id", None)
        or metadata.get("chunk_id")
        or metadata.get("document_id")
        or metadata.get("citation_label")
        or node.node.text[:80]
    )
