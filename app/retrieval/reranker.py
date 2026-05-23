"""
Reranker: deterministic metadata-aware ranking for retrieved nodes.
"""

import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache

from app.core.config import settings
from app.retrieval.cross_encoder_reranker import CrossEncoderSemanticReranker

logger = logging.getLogger(__name__)

# version_rank is month-encoded as year*100+month (e.g. 202603) or
# quarter-encoded as year*10+quarter (e.g. 20261). RANK_SCALE is the upper
# bound so we can normalize all encoding schemes to [0, 1].
_RANK_SCALE = 203012
_MAX_RANK_CONTRIBUTION = 0.15

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
    "how many",
    "how much",
    "number",
    "number of",
    "count",
    "numbers",
    "metric",
    "metrics",
    "percentage",
    "percent",
    "value",
    "values",
)
COUNT_QUERY_TERMS = ("how many", "number of", "count", "total")
COUNT_BREAKDOWN_TERMS = (
    "top customers",
    "customer type",
    "% by arr",
    "annualized recurring revenue",
    "locations",
    "location",
    "rank",
    "investment grade",
)
COUNT_SUMMARY_CHUNK_TYPES = {
    "page_card",
    "visual_proxy_text",
    "chart_context",
    "chart_data_points",
    "figure_artifact",
    "full_table",
    "table_segment",
    "table_summary_text",
}
METRIC_STOPWORDS = {
    "does",
    "have",
    "has",
    "many",
    "much",
    "what",
    "which",
    "with",
    "from",
    "that",
    "this",
    "digital",
    "realty",
}
METRIC_VALUE_PATTERN = re.compile(
    r"\b\$?\d[\d,]*(?:\.\d+)?\s*(?:[+%x]|bn|m|billion|million)?\s+"
    r"(?:global\s+)?(?P<subject>[a-z][a-z-]+)s?\b",
    re.IGNORECASE,
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
    temporal_context = _resolve_temporal_context(source_nodes)
    scored = []
    for node in source_nodes:
        semantic_score = _safe_float(node.score, 0.0)
        metadata = node.node.metadata or {}
        node_id = str(getattr(node.node, "node_id", None) or id(node))

        temporal_adjustment = _temporal_adjustment(
            metadata,
            now,
            prefer_latest,
            node_id in temporal_context.implicit_current,
            node_id in temporal_context.superseded,
        )
        authority_adjustment = _authority_adjustment(metadata)
        structural_adjustment = _structural_adjustment(
            metadata, query, node.node.text or ""
        )
        combined_score = (
            semantic_score
            + temporal_adjustment
            + authority_adjustment
            + structural_adjustment
        )
        node.score = combined_score
        scored.append((combined_score, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    metadata_ranked_nodes = [node for _, node in scored]

    if settings.ENABLE_CROSS_ENCODER_RERANKING and query:
        semantic_reranker = _get_cross_encoder_reranker()
        result = semantic_reranker.rerank(
            query=query,
            nodes=metadata_ranked_nodes,
            top_k=top_k,
        )
        logger.info(
            "Reranked %d nodes with cross-encoder %s, returning top %d",
            len(source_nodes),
            semantic_reranker.model_name,
            len(result),
        )
        return result

    result = metadata_ranked_nodes[:top_k]

    logger.info("Reranked %d nodes, returning top %d", len(source_nodes), len(result))
    return result


@lru_cache(maxsize=1)
def _get_cross_encoder_reranker() -> CrossEncoderSemanticReranker:
    return CrossEncoderSemanticReranker(
        model_name=settings.CROSS_ENCODER_RERANK_MODEL,
        fallback_model_name=settings.CROSS_ENCODER_RERANK_FALLBACK_MODEL,
        hf_api_token=settings.hf_api_token,
        device=settings.CROSS_ENCODER_RERANK_DEVICE,
        trust_remote_code=settings.CROSS_ENCODER_RERANK_TRUST_REMOTE_CODE,
    )


class _TemporalContext:
    def __init__(self) -> None:
        self.implicit_current: set[str] = set()
        self.superseded: set[str] = set()


def _resolve_temporal_context(nodes: list) -> _TemporalContext:
    """Return candidate-local version signals for latest-version ranking.

    When a version group has no explicit is_current=True document (e.g. neither
    filename contains 'final'/'latest'), every node from the highest-ranked
    version in that group is promoted to implicit current so recency preference
    works across the whole document, not just one arbitrary chunk.
    """
    context = _TemporalContext()
    groups: dict[str, list] = defaultdict(list)
    for node in nodes:
        meta = node.node.metadata or {}
        group = meta.get("document_version_group")
        if group:
            groups[group].append(node)

    for group_nodes in groups.values():
        ranked_nodes = [
            node
            for node in group_nodes
            if _safe_int((node.node.metadata or {}).get("version_rank"), 0) > 0
        ]
        if not ranked_nodes:
            continue

        explicit_current_nodes = [
            node
            for node in group_nodes
            if _safe_bool((node.node.metadata or {}).get("is_current"), False)
            is True
        ]
        if explicit_current_nodes:
            current_rank = max(
                _safe_int((node.node.metadata or {}).get("version_rank"), 0)
                for node in explicit_current_nodes
            )
            for node in ranked_nodes:
                node_id = str(getattr(node.node, "node_id", None) or id(node))
                rank = _safe_int((node.node.metadata or {}).get("version_rank"), 0)
                if rank < current_rank:
                    context.superseded.add(node_id)
            continue

        max_rank = max(
            _safe_int((node.node.metadata or {}).get("version_rank"), 0)
            for node in ranked_nodes
        )
        for node in ranked_nodes:
            node_id = str(getattr(node.node, "node_id", None) or id(node))
            rank = _safe_int((node.node.metadata or {}).get("version_rank"), 0)
            if rank == max_rank:
                context.implicit_current.add(node_id)
            elif rank < max_rank:
                context.superseded.add(node_id)

    return context


def _temporal_adjustment(
    metadata: dict,
    now: datetime,
    prefer_latest: bool = True,
    is_implicit_current: bool = False,
    is_superseded: bool = False,
) -> float:
    adjustment = 0.0

    # Explicit is_current wins; fall back to implicit promotion from version group.
    is_current = _safe_bool(metadata.get("is_current"), None)
    if is_current is True or is_implicit_current:
        adjustment += 0.1 if prefer_latest else 0.01
    elif is_current is False and prefer_latest:
        adjustment -= 0.05

    if is_superseded and prefer_latest:
        adjustment -= 0.12

    # Normalize version_rank to [0, 1] before scaling so month-encoded ranks
    # (e.g. 202512, 202603) retain their relative ordering instead of all being
    # clamped to the same cap.
    version_rank = _safe_int(metadata.get("version_rank"), 0)
    if version_rank > 0:
        normalized = min(version_rank / _RANK_SCALE, 1.0)
        scale = 1.0 if prefer_latest else 0.25
        adjustment += normalized * _MAX_RANK_CONTRIBUTION * scale

    effective_from = _parse_date(metadata.get("effective_from"))
    effective_to = _parse_date(metadata.get("effective_to"))
    if effective_from and effective_to:
        if effective_from <= now <= effective_to:
            adjustment += 0.08
        elif (
            now > effective_to
            and prefer_latest
            and not (is_current is True or is_implicit_current)
        ):
            # Graduated penalty: older expiry = larger penalty, capped at -0.10.
            months_overdue = (now - effective_to).days / 30.0
            adjustment -= min(months_overdue * 0.02, 0.10)
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


def _structural_adjustment(metadata: dict, query: str | None, text: str = "") -> float:
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
    if _is_count_metric_query(normalized_query):
        adjustment += _count_metric_adjustment(metadata, normalized_query, text)
    if wants_structured and (is_table_chunk or is_chart_chunk):
        adjustment += 0.06

    return max(min(adjustment, 0.6), -0.25)


def _is_count_metric_query(normalized_query: str) -> bool:
    return any(term in normalized_query for term in COUNT_QUERY_TERMS)


def _count_metric_adjustment(metadata: dict, normalized_query: str, text: str) -> float:
    if not text:
        return 0.0

    query_subjects = _metric_query_subjects(normalized_query)
    if not query_subjects:
        return 0.0

    normalized_text = text.lower()
    direct_subject_match = any(
        _normalize_metric_subject(match.group("subject")) in query_subjects
        for match in METRIC_VALUE_PATTERN.finditer(normalized_text)
    )

    chunk_type = str(metadata.get("chunk_type") or "")
    adjustment = 0.0
    if direct_subject_match:
        adjustment += 0.34
        if chunk_type == "body_text":
            adjustment += 0.12
        if "global customers" in normalized_text:
            adjustment -= 0.06
    if chunk_type in COUNT_SUMMARY_CHUNK_TYPES and any(
        phrase in normalized_text for phrase in COUNT_BREAKDOWN_TERMS
    ):
        adjustment -= 0.22
    elif any(phrase in normalized_text for phrase in COUNT_BREAKDOWN_TERMS):
        adjustment -= 0.12
    return adjustment


def _metric_query_subjects(normalized_query: str) -> set[str]:
    subjects: set[str] = set()
    for token in re.findall(r"[a-z][a-z-]+", normalized_query):
        if len(token) < 4 or token in METRIC_STOPWORDS:
            continue
        subjects.add(_normalize_metric_subject(token))
    return subjects


def _normalize_metric_subject(token: str) -> str:
    normalized = token.lower().strip("-")
    if normalized.endswith("ies") and len(normalized) > 4:
        return f"{normalized[:-3]}y"
    if normalized.endswith("s") and len(normalized) > 3:
        return normalized[:-1]
    return normalized


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
