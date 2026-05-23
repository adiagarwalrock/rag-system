"""Canonical retrieval metadata normalization for parser outputs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from app.ingestion.parser.custom.pdf_pipeline.helpers import (
    has_chart_signals,
    has_numeric_data,
    has_table_signals,
    numeric_density,
)

TABLE_CHUNK_TYPES = {"full_table", "table_segment", "table_summary_text"}
CHART_CHUNK_TYPES = {
    "figure_artifact",
    "chart_context",
    "visual_proxy_text",
    "chart_data_points",
}
VISUAL_CHUNK_TYPES = CHART_CHUNK_TYPES | {"page_card"}

CANONICAL_RETRIEVAL_METADATA_FIELDS = (
    "document_id",
    "document_name",
    "file_name",
    "source_file",
    "client_id",
    "client_name",
    "file_type",
    "ingestion_job_id",
    "chunk_id",
    "chunk_type",
    "parser_name",
    "parser_version",
    "page_num",
    "page_nums",
    "slide_num",
    "section_path",
    "citation_label",
    "contains_numeric_data",
    "numeric_density",
    "table_detected",
    "chart_detected",
    "asset_refs",
    "source_artifact_type",
    "figure_type",
    "document_version_group",
    "version_label",
    "version_rank",
    "effective_from",
    "effective_to",
    "published_at",
    "is_current",
    "document_date",
    "as_of_date",
    "metric_basis",
)


def normalize_retrieval_metadata(
    metadata: dict[str, Any] | None,
    *,
    text: str = "",
    document_metadata: dict[str, Any] | None = None,
    version_info: dict[str, Any] | None = None,
    source_file: str | None = None,
    chunk_index: int | None = None,
) -> dict[str, Any]:
    """Return parser-agnostic metadata consumed by retrieval and reranking."""
    normalized: dict[str, Any] = {}
    if document_metadata:
        normalized.update(document_metadata)
    if metadata:
        normalized.update(metadata)
    if version_info:
        _apply_version_info(normalized, version_info)

    resolved_source = _first_text(
        source_file,
        normalized.get("source_file"),
        normalized.get("file_name"),
        normalized.get("document_name"),
    )
    if resolved_source:
        normalized["source_file"] = resolved_source
        normalized.setdefault("file_name", resolved_source)
        normalized.setdefault("document_name", resolved_source)

    normalized["chunk_id"] = _first_text(normalized.get("chunk_id")) or str(uuid.uuid4())
    normalized["chunk_type"] = _first_text(normalized.get("chunk_type")) or "text"
    normalized["parser_name"] = _first_text(normalized.get("parser_name")) or "unknown"
    normalized["parser_version"] = _first_text(normalized.get("parser_version"))
    normalized["section_path"] = _first_text(normalized.get("section_path")) or ""

    _normalize_location_fields(normalized)
    _normalize_retrieval_signals(normalized, text)
    _normalize_temporal_fields(normalized)

    if not normalized.get("citation_label"):
        normalized["citation_label"] = build_citation_label(
            normalized,
            source_file=resolved_source or "Source",
            chunk_index=chunk_index,
        )
    _ensure_canonical_defaults(normalized)

    return normalized


def build_citation_label(
    metadata: dict[str, Any],
    *,
    source_file: str,
    chunk_index: int | None = None,
) -> str:
    parts = [source_file]
    page_num = _safe_int(metadata.get("page_num"))
    slide_num = _safe_int(metadata.get("slide_num"))
    if page_num:
        parts.append(f"p.{page_num}")
    if slide_num:
        parts.append(f"slide {slide_num}")
    if chunk_index is not None:
        parts.append(f"chunk {chunk_index}")
    return " - ".join(parts)


def _apply_version_info(metadata: dict[str, Any], version_info: dict[str, Any]) -> None:
    mapping = {
        "document_version_group": version_info.get("version_group"),
        "version_label": version_info.get("version_label"),
        "version_rank": version_info.get("version_rank", 0),
        "effective_from": version_info.get("effective_from"),
        "effective_to": version_info.get("effective_to"),
        "published_at": version_info.get("published_at"),
        "is_current": version_info.get("is_current", False),
    }
    for key, value in mapping.items():
        if value is not None:
            metadata[key] = value


def _normalize_location_fields(metadata: dict[str, Any]) -> None:
    page_nums = _coerce_int_list(metadata.get("page_nums"))
    page_num = _safe_int(metadata.get("page_num"))
    slide_num = _safe_int(metadata.get("slide_num"))
    if not page_nums:
        if page_num:
            page_nums = [page_num]
        elif slide_num:
            page_nums = [slide_num]
    if page_nums:
        metadata["page_nums"] = page_nums
        metadata["page_num"] = page_num or page_nums[0]
    else:
        metadata["page_nums"] = []
        metadata["page_num"] = page_num
    metadata["slide_num"] = slide_num


def _normalize_retrieval_signals(metadata: dict[str, Any], text: str) -> None:
    chunk_type = str(metadata.get("chunk_type") or "")
    metadata["table_detected"] = _safe_bool(
        metadata.get("table_detected"),
        chunk_type in TABLE_CHUNK_TYPES or has_table_signals(text),
    )
    metadata["chart_detected"] = _safe_bool(
        metadata.get("chart_detected"),
        chunk_type in CHART_CHUNK_TYPES or has_chart_signals(text),
    )
    metadata["contains_numeric_data"] = _safe_bool(
        metadata.get("contains_numeric_data"),
        has_numeric_data(text),
    )
    metadata["numeric_density"] = max(
        _safe_float(metadata.get("numeric_density"), 0.0),
        numeric_density(text) if text else 0.0,
    )
    metadata["asset_refs"] = _coerce_string_list(metadata.get("asset_refs"))
    metadata["source_artifact_type"] = (
        _first_text(metadata.get("source_artifact_type")) or chunk_type or "text"
    )
    metadata["figure_type"] = _first_text(metadata.get("figure_type"))


def _normalize_temporal_fields(metadata: dict[str, Any]) -> None:
    metadata["version_rank"] = _safe_int(metadata.get("version_rank")) or 0
    metadata["is_current"] = _safe_bool(metadata.get("is_current"), False)
    for key in ("effective_from", "effective_to", "published_at"):
        metadata[key] = _isoformat_or_none(metadata.get(key))
    for key in ("document_date", "as_of_date", "metric_basis"):
        metadata[key] = _first_text(metadata.get(key))


def _ensure_canonical_defaults(metadata: dict[str, Any]) -> None:
    defaults = {
        "document_id": None,
        "document_name": None,
        "file_name": None,
        "source_file": None,
        "client_id": None,
        "client_name": None,
        "file_type": None,
        "ingestion_job_id": None,
        "slide_num": None,
        "figure_type": None,
        "document_version_group": None,
        "version_label": None,
        "published_at": None,
        "effective_from": None,
        "effective_to": None,
        "document_date": None,
        "as_of_date": None,
        "metric_basis": None,
    }
    for key, default in defaults.items():
        metadata.setdefault(key, default)
    for key in CANONICAL_RETRIEVAL_METADATA_FIELDS:
        metadata.setdefault(key, None)


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, list | tuple | set) else [value]
    coerced: list[int] = []
    for item in values:
        parsed = _safe_int(item)
        if parsed is not None:
            coerced.append(parsed)
    return coerced


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list | tuple | set) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        cleaned = _first_text(item)
        if not cleaned or cleaned in seen:
            continue
        result.append(cleaned)
        seen.add(cleaned)
    return result


def _isoformat_or_none(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
