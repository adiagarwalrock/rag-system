"""
Reranker: deterministic metadata-aware ranking for retrieved nodes.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

TABLE_QUERY_TERMS = (
    "table",
    "row",
    "rows",
    "column",
    "columns",
    "matrix",
    "tabular",
    "breakdown",
    "top ",
    "top-",
    "portfolio composition",
    "market mix",
)
CHART_QUERY_TERMS = (
    "chart",
    "graph",
    "plot",
    "trend",
    "map",
    "diagram",
    "figure",
    "legend",
    "infographic",
    "pie",
    "bar",
    "line",
)
IMAGE_QUERY_TERMS = ("image", "images", "screenshot", "screenshots", "visual")
NUMERIC_QUERY_TERMS = (
    "number",
    "numbers",
    "metric",
    "metrics",
    "percentage",
    "percent",
    "value",
    "values",
)
# Terms that indicate a question is asking for headline / overview statistics
_FACTUAL_LOOKUP_TERMS = (
    "quick facts",
    "fast facts",
    "headline stats",
    "at a glance",
    "key stats",
    "key facts",
    "what are",
    "how many",
    "what is",
    "what was",
    "what does",
    "how much",
    "overview",
    "summary",
)


def rerank_nodes(
    source_nodes: list,
    top_k: int = 10,
    prefer_latest: bool = True,
    query: str | None = None,
) -> list:
    """
    Rank retrieved nodes by combining semantic similarity with one temporal pass.

    Args:
        source_nodes: LlamaIndex NodeWithScore objects
        top_k: Number of top results to return
        prefer_latest: If True, apply stronger recency/current-version preference
        query: Optional user query for intent-aware structural boosts

    Returns:
        Reranked list of source nodes.
    """
    now = datetime.now(timezone.utc)
    scored = []
    for node in source_nodes:
        semantic_score = _safe_float(node.score, 0.0)
        metadata = node.node.metadata or {}

        temporal_adjustment = _temporal_adjustment(metadata, now, prefer_latest)
        authority_adjustment = _authority_adjustment(metadata)
        structural_adjustment = _structural_adjustment(metadata, query)
        combined_score = (
            semantic_score
            + temporal_adjustment
            + authority_adjustment
            + structural_adjustment
        )
        node.score = combined_score
        scored.append((combined_score, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [node for _, node in scored[:top_k]]

    _apply_version_consistency_adjustment(result, query or "")

    logger.info("Reranked %d nodes, returning top %d", len(source_nodes), len(result))
    return result


def _temporal_adjustment(
    metadata: dict, now: datetime, prefer_latest: bool = True
) -> float:
    adjustment = 0.0

    is_current = _safe_bool(metadata.get("is_current"), None)
    if is_current is True:
        adjustment += 0.1 if prefer_latest else 0.01
    elif is_current is False and prefer_latest:
        adjustment -= 0.05

    version_rank = _safe_int(metadata.get("version_rank"), 0)
    if version_rank > 0:
        weight = 0.001 if prefer_latest else 0.00025
        adjustment += min(version_rank, 100) * weight

    effective_from = _parse_date(metadata.get("effective_from"))
    effective_to = _parse_date(metadata.get("effective_to"))
    if effective_from and effective_to:
        if effective_from <= now <= effective_to:
            adjustment += 0.08
        elif now > effective_to and prefer_latest:
            adjustment -= 0.05
        elif now < effective_from and prefer_latest:
            adjustment -= 0.03

    published_at = _parse_date(metadata.get("published_at"))
    if published_at and prefer_latest:
        days_old = (now - published_at).days
        if days_old < 90:
            adjustment += 0.04
        elif days_old < 365:
            adjustment += 0.015
        elif days_old > 730:
            adjustment -= 0.02

    return adjustment


def _authority_adjustment(metadata: dict) -> float:
    """Only apply authority when explicit non-neutral metadata is present."""
    if "authority_score" not in metadata:
        return 0.0

    authority = _safe_float(metadata.get("authority_score"), None)
    if authority is None or authority == 1.0:
        return 0.0

    normalized = max(min(authority - 1.0, 1.0), -1.0)
    return normalized * 0.05


def _structural_adjustment(metadata: dict, query: str | None) -> float:
    if not query:
        return 0.0

    normalized_query = query.lower()
    chunk_type = str(metadata.get("chunk_type") or "")
    figure_type = str(metadata.get("figure_type") or "")

    wants_table = any(term in normalized_query for term in TABLE_QUERY_TERMS)
    wants_chart = any(term in normalized_query for term in CHART_QUERY_TERMS)
    wants_image = any(term in normalized_query for term in IMAGE_QUERY_TERMS)
    wants_numeric = any(term in normalized_query for term in NUMERIC_QUERY_TERMS)
    wants_structured = wants_table or wants_chart or "legend" in normalized_query

    is_table_chunk = chunk_type in {
        "full_table",
        "table_segment",
        "table_summary_text",
    } or _safe_bool(metadata.get("table_detected"), False)
    is_chart_chunk = (
        chunk_type
        in {
            "figure_artifact",
            "chart_context",
            "chart_data_points",
            "visual_proxy_text",
        }
        or _safe_bool(metadata.get("chart_detected"), False)
        or figure_type in {"chart", "diagram", "infographic"}
    )
    is_reasoning_chunk = chunk_type.startswith("reasoning_")

    adjustment = 0.0
    if wants_table and is_table_chunk:
        adjustment += 0.12
    if wants_chart and is_chart_chunk:
        adjustment += 0.12
    if (wants_table or wants_chart) and is_reasoning_chunk:
        adjustment += 0.05
    if wants_image and bool(metadata.get("asset_refs")):
        adjustment += 0.06
    if wants_numeric and _safe_bool(metadata.get("contains_numeric_data"), False):
        adjustment += 0.03
    if wants_structured and (is_table_chunk or is_chart_chunk):
        adjustment += 0.06

    # Boost overview/summary slides for factual lookup queries
    slide_purpose = str(metadata.get("slide_purpose") or "")
    if slide_purpose == "overview_stats":
        is_factual_lookup = any(
            term in normalized_query for term in _FACTUAL_LOOKUP_TERMS
        )
        if is_factual_lookup:
            adjustment += 0.06

    return min(adjustment, 0.26)


def _parse_date(val) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None
    return None


def _safe_float(value, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


_COMPARISON_TERMS = (
    "compare",
    "comparison",
    "difference",
    "versus",
    "vs",
    "between",
    "changed",
    "changes",
)
_VERSION_PENALTY = 0.03


def _apply_version_consistency_adjustment(
    nodes: list,
    query: str,
) -> None:
    """Penalize non-dominant versions of the same document for non-comparison queries."""
    if not nodes or not query:
        return
    normalized = query.lower()
    if any(term in normalized for term in _COMPARISON_TERMS):
        return

    doc_version_scores: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for node in nodes:
        metadata = node.node.metadata or {}
        doc_id = metadata.get("document_id") or metadata.get("document_name") or ""
        version = metadata.get("version_label") or ""
        if doc_id and version:
            doc_version_scores[doc_id][version] += _safe_float(node.score, 0.0) or 0.0

    dominant_version: dict[str, str] = {}
    for doc_id, version_scores in doc_version_scores.items():
        if len(version_scores) < 2:
            continue
        dominant_version[doc_id] = max(version_scores, key=version_scores.get)

    if not dominant_version:
        return

    for node in nodes:
        metadata = node.node.metadata or {}
        doc_id = metadata.get("document_id") or metadata.get("document_name") or ""
        version = metadata.get("version_label") or ""
        if (
            doc_id in dominant_version
            and version
            and version != dominant_version[doc_id]
        ):
            node.score = (_safe_float(node.score, 0.0) or 0.0) - _VERSION_PENALTY
