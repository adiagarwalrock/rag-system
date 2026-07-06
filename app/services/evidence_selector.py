"""
Evidence selection, node classification, and diversity helpers.

Extracted from retriever.py to keep retrieval orchestration separate
from evidence-level logic.
"""

import logging
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REASONING_CHUNK_TYPES = {
    "reasoning_table",
    "reasoning_chart",
    "reasoning_figure",
    "reasoning_page",
}
REASONING_PRIORITY_TERMS = (
    "chart",
    "charts",
    "graph",
    "graphs",
    "table",
    "tables",
    "figure",
    "figures",
    "diagram",
    "diagrams",
    "image",
    "images",
    "screenshot",
    "screenshots",
    "visual",
    "trend",
    "trends",
    "why",
    "reason",
    "analysis",
    "explain",
)
COMPARATIVE_TERMS = (
    "compare",
    "comparison",
    "difference",
    "change",
    "changes",
    "changed",
    "conflict",
    "conflicting",
    "contradict",
    "versus",
    "vs",
    "between",
)
VISUAL_QUERY_TERMS = (
    "image",
    "images",
    "figure",
    "figures",
    "chart",
    "charts",
    "graph",
    "graphs",
    "diagram",
    "diagrams",
    "map",
    "maps",
    "screenshot",
    "screenshots",
    "visual",
)
TABLE_EVIDENCE_TERMS = (
    "table",
    "matrix",
    "tabular",
    "top ",
    "top-",
    "breakdown",
    "market mix",
    "portfolio composition",
)
CHART_EVIDENCE_TERMS = (
    "chart",
    "graph",
    "plot",
    "legend",
    "infographic",
    "pie",
    "trend",
    "line",
    "bar",
)
NUMERIC_INTENT_TERMS = (
    "how much",
    "what percent",
    "percentage",
    "difference between",
    "increase",
    "decrease",
    "grew",
    "declined",
    "revenue",
    "margin",
    "ratio",
    "total",
    "average",
)
DATA_RETRIEVAL_TERMS = (
    "list all",
    "what are the",
    "ranking",
    "how many",
)
TIME_ANCHORED_TERMS = (
    "as of",
    "q1",
    "q2",
    "q3",
    "q4",
    "fy",
    "fiscal",
    "expected close",
)
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMERIC_DENSITY_PATTERN = re.compile(r"\b\d[\d,.]*\b")
_MAP_LAYOUT_TERMS = (
    "map",
    "floor plan",
    "geographic",
    "footprint",
    "location map",
    "property map",
    "world map",
    "us map",
)


# ---------------------------------------------------------------------------
# Query classifiers
# ---------------------------------------------------------------------------


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


def _is_reasoning_priority_query(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in REASONING_PRIORITY_TERMS)


def _is_visual_or_image_query(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in VISUAL_QUERY_TERMS)


def _is_map_or_layout_query(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in _MAP_LAYOUT_TERMS)


def _is_page_card_node(node: Any) -> bool:
    return str((node.node.metadata or {}).get("chunk_type") or "") == "page_card"


def _is_time_anchored_query(question: str) -> bool:
    normalized = question.lower()
    if YEAR_PATTERN.search(normalized):
        return True
    return any(term in normalized for term in TIME_ANCHORED_TERMS)


def _has_numeric_density(question: str) -> bool:
    """Return True when a query contains 2+ numeric tokens, suggesting tabular data."""
    return len(_NUMERIC_DENSITY_PATTERN.findall(question)) >= 2


def _has_implicit_structured_intent(question: str) -> bool:
    """Return True for queries that imply structured data without explicit keywords."""
    normalized = question.lower()
    return any(term in normalized for term in NUMERIC_INTENT_TERMS) or any(
        term in normalized for term in DATA_RETRIEVAL_TERMS
    )


def _wants_table_evidence(question: str) -> bool:
    normalized = question.lower()
    if any(term in normalized for term in TABLE_EVIDENCE_TERMS):
        return True
    return _has_implicit_structured_intent(question) or _has_numeric_density(question)


def _wants_chart_evidence(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in CHART_EVIDENCE_TERMS)


def _doc_family(node: Any) -> str:
    meta = getattr(node, "node", node).metadata or {}
    return (
        meta.get("document_family")
        or meta.get("document_version_group")
        or meta.get("document_id")
        or ""
    )


def _is_cross_document_synthesis_query(ranked_nodes: list) -> bool:
    """True when the retrieved pool spans at least 3 distinct document families."""
    families = {_doc_family(n) for n in ranked_nodes if _doc_family(n)}
    return len(families) >= 3


# ---------------------------------------------------------------------------
# Node classifiers
# ---------------------------------------------------------------------------


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


def _has_numeric_signal(node: Any) -> bool:
    metadata = node.node.metadata or {}
    raw = metadata.get("contains_numeric_data")
    return _safe_bool(raw, False)


def _node_has_image_assets(node: Any) -> bool:
    metadata = node.node.metadata or {}
    refs = metadata.get("asset_refs")
    if isinstance(refs, str):
        return bool(refs.strip())
    if isinstance(refs, list):
        return any(isinstance(ref, str) and ref.strip() for ref in refs)
    return False


def _is_reasoning_chunk(node: Any) -> bool:
    metadata = node.node.metadata or {}
    return str(metadata.get("chunk_type") or "") in REASONING_CHUNK_TYPES


def _is_table_like_node(node: Any) -> bool:
    metadata = node.node.metadata or {}
    chunk_type = str(metadata.get("chunk_type") or "")
    return chunk_type in {
        "full_table",
        "table_segment",
        "table_summary_text",
        "reasoning_table",
    } or _safe_bool(metadata.get("table_detected"), False)


def _is_chart_like_node(node: Any) -> bool:
    metadata = node.node.metadata or {}
    chunk_type = str(metadata.get("chunk_type") or "")
    figure_type = str(metadata.get("figure_type") or "")
    return (
        chunk_type
        in {
            "figure_artifact",
            "chart_context",
            "chart_data_points",
            "visual_proxy_text",
            "chart_legend_chunk",
            "reasoning_chart",
            "reasoning_figure",
        }
        or _safe_bool(metadata.get("chart_detected"), False)
        or figure_type in {"chart", "diagram", "infographic"}
    )


# ---------------------------------------------------------------------------
# Node key / diversity helpers
# ---------------------------------------------------------------------------


def _node_unique_key(node: Any) -> str:
    metadata = node.node.metadata or {}
    return str(
        getattr(node.node, "node_id", None)
        or metadata.get("chunk_id")
        or metadata.get("document_id")
        or metadata.get("citation_label")
        or node.node.text[:80]
    )


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


def _document_diversity_keys(nodes: list[Any]) -> set[str]:
    keys: set[str] = set()
    for node in nodes:
        metadata = node.node.metadata or {}
        value = (
            metadata.get("document_id")
            or metadata.get("document_name")
            or metadata.get("source_file")
            or getattr(node.node, "node_id", None)
        )
        if value:
            keys.add(str(value))
    return keys


def _chunk_type_counts(nodes: list[Any]) -> dict[str, int]:
    return dict(
        Counter(
            str((node.node.metadata or {}).get("chunk_type") or "unknown")
            for node in nodes
        )
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _build_retrieval_diagnostics(
    ranked_nodes: list[Any],
    evidence_nodes: list[Any],
) -> dict[str, Any]:
    ranked_chunk_types = _chunk_type_counts(ranked_nodes)
    evidence_chunk_types = _chunk_type_counts(evidence_nodes)
    ranked_reasoning = sum(
        count
        for chunk_type, count in ranked_chunk_types.items()
        if chunk_type in REASONING_CHUNK_TYPES
    )
    evidence_reasoning = sum(
        count
        for chunk_type, count in evidence_chunk_types.items()
        if chunk_type in REASONING_CHUNK_TYPES
    )
    ranked_image_count = sum(1 for node in ranked_nodes if _node_has_image_assets(node))
    evidence_image_count = sum(
        1 for node in evidence_nodes if _node_has_image_assets(node)
    )

    return {
        "ranked_document_count": len(_document_diversity_keys(ranked_nodes)),
        "evidence_document_count": len(_document_diversity_keys(evidence_nodes)),
        "ranked_chunk_types": ranked_chunk_types,
        "evidence_chunk_types": evidence_chunk_types,
        "ranked_reasoning_count": ranked_reasoning,
        "evidence_reasoning_count": evidence_reasoning,
        "ranked_image_chunk_count": ranked_image_count,
        "evidence_image_chunk_count": evidence_image_count,
    }


# ---------------------------------------------------------------------------
# Evidence injection / structured evidence guarantees
# ---------------------------------------------------------------------------


def _is_stats_slide_node(node: Any) -> bool:
    meta = node.node.metadata or {}
    return meta.get("slide_purpose") in {"overview_stats", "data_slide"}


def _wants_stats_slide_evidence(question: str) -> bool:
    lowered = question.lower()
    return any(
        term in lowered
        for term in (
            "properties",
            "leased",
            "walt",
            "abr",
            "occupancy",
            "enterprise value",
            "total",
            "portfolio",
            "sq ft",
            "square feet",
            "assets under",
            "aum",
            "noi",
            "ebitda",
            "quick facts",
            "fast facts",
            "at a glance",
        )
    )


def _ensure_structured_evidence(
    *,
    question: str,
    selected_nodes: list[Any],
    ranked_nodes: list[Any],
    evidence_cap: int,
) -> list[Any]:
    want_table = _wants_table_evidence(question)
    want_chart = _wants_chart_evidence(question)
    want_stats = _wants_stats_slide_evidence(question)

    if not want_table and not want_chart and not want_stats:
        return selected_nodes

    selected = list(selected_nodes)
    selected_keys = {_node_unique_key(node) for node in selected}

    anchor_doc_id = None
    if selected and not _is_comparison_or_conflict_query(question):
        anchor_doc_id = (selected[0].node.metadata or {}).get("document_id")

    requirements: list[tuple[bool, Any, str]] = [
        (want_table, _is_table_like_node, "table"),
        (want_chart, _is_chart_like_node, "chart"),
        (want_stats, _is_stats_slide_node, "stats_slide"),
    ]

    for enabled, predicate, requirement_name in requirements:
        if not enabled:
            continue
        if any(predicate(node) for node in selected):
            continue
        candidate = next(
            (
                node
                for node in ranked_nodes
                if predicate(node)
                and _node_unique_key(node) not in selected_keys
                and (
                    not anchor_doc_id
                    or (node.node.metadata or {}).get("document_id") == anchor_doc_id
                )
            ),
            None,
        )
        if candidate is None and anchor_doc_id:
            candidate = next(
                (
                    node
                    for node in ranked_nodes
                    if predicate(node) and _node_unique_key(node) not in selected_keys
                ),
                None,
            )
        if candidate is None:
            logger.debug(
                "Structured evidence requirement unmet (type=%s): no ranked candidate",
                requirement_name,
            )
            continue

        if len(selected) < evidence_cap:
            selected.append(candidate)
            selected_keys.add(_node_unique_key(candidate))
            continue

        replace_idx = next(
            (
                idx
                for idx, node in reversed(list(enumerate(selected)))
                if not predicate(node)
            ),
            None,
        )
        if replace_idx is None:
            continue

        selected_keys.discard(_node_unique_key(selected[replace_idx]))
        selected[replace_idx] = candidate
        selected_keys.add(_node_unique_key(candidate))

    return selected


def _ensure_image_evidence(
    *,
    question: str,
    selected_nodes: list[Any],
    ranked_nodes: list[Any],
    evidence_cap: int,
) -> list[Any]:
    if not _is_visual_or_image_query(question):
        return selected_nodes

    desired_image_nodes = min(2, evidence_cap)
    selected = list(selected_nodes)
    selected_keys = {_node_unique_key(node) for node in selected}
    image_node_count = sum(1 for node in selected if _node_has_image_assets(node))

    for candidate in ranked_nodes:
        if image_node_count >= desired_image_nodes:
            break
        if not _node_has_image_assets(candidate):
            continue
        candidate_key = _node_unique_key(candidate)
        if candidate_key in selected_keys:
            continue

        if len(selected) < evidence_cap:
            selected.append(candidate)
            selected_keys.add(candidate_key)
            image_node_count += 1
            continue

        replace_idx = next(
            (
                idx
                for idx, node in reversed(list(enumerate(selected)))
                if not _node_has_image_assets(node)
            ),
            None,
        )
        if replace_idx is None:
            break

        selected_keys.discard(_node_unique_key(selected[replace_idx]))
        selected[replace_idx] = candidate
        selected_keys.add(candidate_key)
        image_node_count += 1

    # For map/layout queries, also attempt to inject a page_card with a screenshot
    if _is_map_or_layout_query(question):
        for candidate in ranked_nodes:
            if image_node_count >= desired_image_nodes:
                break
            if not _is_page_card_node(candidate) or not _node_has_image_assets(
                candidate
            ):
                continue
            candidate_key = _node_unique_key(candidate)
            if candidate_key in selected_keys:
                continue

            if len(selected) < evidence_cap:
                selected.append(candidate)
                selected_keys.add(candidate_key)
                image_node_count += 1
                continue

            replace_idx = next(
                (
                    idx
                    for idx, node in reversed(list(enumerate(selected)))
                    if not _node_has_image_assets(node)
                ),
                None,
            )
            if replace_idx is None:
                break

            selected_keys.discard(_node_unique_key(selected[replace_idx]))
            selected[replace_idx] = candidate
            selected_keys.add(candidate_key)
            image_node_count += 1

    return selected
