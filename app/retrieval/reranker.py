"""
Reranker: deterministic metadata-aware ranking for retrieved nodes.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def rerank_nodes(
    source_nodes: list, top_k: int = 10, prefer_latest: bool = True
) -> list:
    """
    Rank retrieved nodes by combining semantic similarity with one temporal pass.

    Args:
        source_nodes: LlamaIndex NodeWithScore objects
        top_k: Number of top results to return
        prefer_latest: If True, apply stronger recency/current-version preference

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
        combined_score = semantic_score + temporal_adjustment + authority_adjustment
        node.score = combined_score
        scored.append((combined_score, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [node for _, node in scored[:top_k]]

    logger.info("Reranked %d nodes, returning top %d", len(source_nodes), len(result))
    return result


def _temporal_adjustment(
    metadata: dict, now: datetime, prefer_latest: bool = True
) -> float:
    adjustment = 0.0

    is_current = _safe_bool(metadata.get("is_current"), None)
    if is_current is True:
        adjustment += 0.1
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
