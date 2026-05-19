"""
Reranker: cross-encoder-based ranking with metadata-aware adjustments.

Scoring formula:
  combined = 0.4 * semantic_score + 0.5 * ce_score_norm + temporal + authority

where ce_score_norm = sigmoid(raw cross-encoder logit), mapping to [0, 1].
Temporal and version-consistency adjustments encode business rules (recency,
document supersession) that a general cross-encoder doesn't know about.
"""

import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache

logger = logging.getLogger(__name__)

# Terms that indicate a comparison query — used by version-consistency logic only.
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

# Score weights
_SEMANTIC_WEIGHT = 0.4
_CE_WEIGHT = 0.5


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    from app.core.config import settings

    logger.info("Loading cross-encoder model: %s", settings.RERANKER_MODEL)
    return TextCrossEncoder(model_name=settings.RERANKER_MODEL)


def _cross_encoder_scores(query: str, nodes: list) -> list[float]:
    encoder = _get_cross_encoder()
    texts = [node.node.get_content() or "" for node in nodes]
    raw_scores = list(encoder.rerank(query, texts))
    return [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]


def rerank_nodes(
    source_nodes: list,
    top_k: int = 15,
    prefer_latest: bool = True,
    query: str | None = None,
) -> list:
    """
    Rank retrieved nodes by combining semantic similarity, cross-encoder
    relevance, and metadata-based temporal/authority adjustments.

    Args:
        source_nodes: LlamaIndex NodeWithScore objects
        top_k: Number of top results to return
        prefer_latest: If True, apply stronger recency/current-version preference
        query: Optional user query for cross-encoder scoring

    Returns:
        Reranked list of source nodes.
    """
    now = datetime.now(timezone.utc)

    ce_scores = (
        _cross_encoder_scores(query, source_nodes)
        if query and source_nodes
        else [0.5] * len(source_nodes)
    )

    scored = []
    for node, ce_score in zip(source_nodes, ce_scores):
        semantic_score = _safe_float(node.score, 0.0)
        metadata = node.node.metadata or {}
        temporal = _temporal_adjustment(metadata, now, prefer_latest)
        authority = _authority_adjustment(metadata)
        combined = (
            _SEMANTIC_WEIGHT * semantic_score
            + _CE_WEIGHT * ce_score
            + temporal
            + authority
        )
        node.score = combined
        scored.append((combined, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [node for _, node in scored[:top_k]]

    _apply_version_consistency_adjustment(result, query or "")

    logger.info(
        "Reranked %d nodes via cross-encoder, returning top %d",
        len(source_nodes),
        len(result),
    )
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


def _apply_version_consistency_adjustment(
    nodes: list,
    query: str,
) -> None:
    """Penalize non-dominant versions of the same document for non-comparison queries."""  # noqa: E501
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

    doc_chunk_counts: dict[str, int] = {}
    for node in nodes:
        meta = node.node.metadata or {}
        doc_id = meta.get("document_id") or meta.get("document_name") or ""
        if doc_id:
            doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1

    for node in nodes:
        metadata = node.node.metadata or {}
        doc_id = metadata.get("document_id") or metadata.get("document_name") or ""
        version = metadata.get("version_label") or ""
        if (
            doc_id in dominant_version
            and version
            and version != dominant_version[doc_id]
        ):
            if doc_chunk_counts.get(doc_id, 0) <= 1:
                continue
            node.score = (_safe_float(node.score, 0.0) or 0.0) - _VERSION_PENALTY
