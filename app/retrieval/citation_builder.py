"""
Citation builder: constructs structured citations from retrieved source nodes.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_TEXT_LIMIT = 1400
RICH_TEXT_LIMIT = 2600
RICH_CHUNK_TYPES = {
    "full_table",
    "table_segment",
    "table_summary_text",
    "figure_artifact",
    "chart_context",
    "chart_data_points",
    "visual_proxy_text",
    "reasoning_table",
    "reasoning_chart",
    "reasoning_figure",
    "reasoning_page",
}
ENRICHED_METADATA_FIELDS = (
    "key_chart_facts",
    "approx_datapoints",
    "trend_summary",
    "units",
    "as_of_date",
    "document_date",
    "metric_basis",
    "claims",
    "evidence_refs",
    "llm_page_summary",
    "llm_caption",
)


def build_citations(source_nodes: list) -> List[Dict[str, Any]]:
    """
    Build structured citation objects from retrieved source nodes.

    Each citation includes document name, page/slide reference,
    version info, relevance score, and a text snippet.

    Args:
        source_nodes: List of NodeWithScore objects

    Returns:
        List of citation dicts.
    """
    citations = []
    seen_asset_signatures: set[str] = set()

    for i, node in enumerate(source_nodes):
        metadata = node.node.metadata or {}
        text = node.node.text or ""
        chunk_type = metadata.get("chunk_type", "text")
        text_limit = _text_limit_for_chunk(chunk_type)
        asset_refs = _coerce_asset_refs(
            metadata.get("asset_refs"),
            artifact_bundle_path=metadata.get("artifact_bundle_path"),
            global_seen_signatures=seen_asset_signatures,
        )

        citation = {
            "rank": i + 1,
            "score": round(node.score, 4) if node.score else None,
            "text": text[:text_limit],
            "vector_node_id": getattr(node.node, "node_id", None),
            "document_id": metadata.get("document_id"),
            "document_name": metadata.get("document_name")
            or metadata.get("file_name")
            or metadata.get("source_file")
            or "Unknown",
            "source_file": metadata.get("source_file", ""),
            "page_num": metadata.get("page_num"),
            "slide_num": metadata.get("slide_num"),
            "section_title": metadata.get("section_title"),
            "chunk_type": chunk_type,
            "source_artifact_type": metadata.get("source_artifact_type"),
            "source_artifact_id": metadata.get("source_artifact_id"),
            "artifact_bundle_path": metadata.get("artifact_bundle_path"),
            "version_label": metadata.get("version_label"),
            "version_group": metadata.get("document_version_group"),
            "effective_from": metadata.get("effective_from"),
            "effective_to": metadata.get("effective_to"),
            "document_date": metadata.get("document_date"),
            "as_of_date": metadata.get("as_of_date"),
            "metric_basis": metadata.get("metric_basis"),
            "citation_label": metadata.get(
                "citation_label", _build_fallback_label(metadata, i + 1)
            ),
            "authority_score": metadata.get("authority_score"),
            "asset_refs": asset_refs,
            "image_assets": build_image_assets(
                asset_refs=asset_refs,
                citation_metadata=metadata,
                document_name=metadata.get("document_name")
                or metadata.get("file_name")
                or metadata.get("source_file")
                or "Unknown",
            ),
            "has_image_assets": bool(asset_refs),
            "figure_type": metadata.get("figure_type"),
            "chart_type": metadata.get("chart_type"),
            "table_id": metadata.get("table_id"),
            "enriched_metadata": _extract_enriched_metadata(metadata),
        }
        citations.append(citation)

    logger.info("Built %d citations from source nodes", len(citations))
    return citations


def _build_fallback_label(metadata: dict, rank: int) -> str:
    """Build a fallback citation label when none exists."""
    parts = []
    doc_name = (
        metadata.get("document_name") or metadata.get("source_file") or f"Source {rank}"
    )
    parts.append(doc_name)

    if metadata.get("page_num"):
        parts.append(f"p.{metadata['page_num']}")
    if metadata.get("slide_num"):
        parts.append(f"slide {metadata['slide_num']}")
    if metadata.get("version_label"):
        parts.append(f"({metadata['version_label']})")

    return " — ".join(parts)


def format_citations_for_prompt(citations: List[Dict[str, Any]]) -> str:
    """
    Format citations into a text block for inclusion in the LLM prompt,
    so the answer can reference sources by label.
    """
    lines = []
    for c in citations:
        label = c.get("citation_label", f"Source {c['rank']}")
        text_preview = c["text"][:450]
        lines.append(f"[{label}]: {text_preview}")
    return "\n\n".join(lines)


def _text_limit_for_chunk(chunk_type: str) -> int:
    return RICH_TEXT_LIMIT if chunk_type in RICH_CHUNK_TYPES else DEFAULT_TEXT_LIMIT


def _extract_enriched_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    enriched: dict[str, Any] = {}
    for field_name in ENRICHED_METADATA_FIELDS:
        value = metadata.get(field_name)
        if _has_enriched_value(value):
            enriched[field_name] = _normalize_enriched_value(value)
    return enriched


def _has_enriched_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _normalize_enriched_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (tuple, set)):
        return list(value)
    return value


def format_enriched_metadata_for_prompt(enriched_metadata: dict[str, Any]) -> str:
    """Render enriched parser facts for answer synthesis prompts."""
    if not enriched_metadata:
        return ""

    lines: list[str] = []
    for field_name, value in enriched_metadata.items():
        formatted = _format_metadata_value(value)
        if formatted:
            lines.append(f"- {field_name}: {formatted}")
    return "\n".join(lines)


def _format_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        truncated = len(value) > 8
        parts = [_format_metadata_value(item) for item in value[:8]]
        result = "; ".join(part for part in parts if part)
        return f"{result}; [+{len(value) - 8} more]" if truncated else result
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)[:1200]
        except TypeError:
            return str(value)[:1200]
    return str(value)


def _coerce_asset_refs(
    raw_refs: Any,
    artifact_bundle_path: str | None = None,
    global_seen_signatures: set[str] | None = None,
) -> list[str]:
    if isinstance(raw_refs, str):
        refs = [raw_refs]
    elif isinstance(raw_refs, list):
        refs = raw_refs
    else:
        return []

    normalized: list[str] = []
    local_signatures: set[str] = set()
    for ref in refs:
        if not isinstance(ref, str):
            continue
        cleaned = ref.strip()
        if not cleaned:
            continue
        normalized_ref = _normalize_asset_ref(cleaned, artifact_bundle_path)
        signature = _asset_signature(normalized_ref)
        if signature in local_signatures:
            continue
        if global_seen_signatures is not None and signature in global_seen_signatures:
            continue
        local_signatures.add(signature)
        if global_seen_signatures is not None:
            global_seen_signatures.add(signature)
        normalized.append(normalized_ref)

    return normalized


def _normalize_asset_ref(ref: str, artifact_bundle_path: str | None) -> str:
    raw_path = Path(ref).expanduser()
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        if artifact_bundle_path:
            candidates.append(Path(artifact_bundle_path).expanduser() / raw_path)
        candidates.append(raw_path)
        candidates.append(Path.cwd() / raw_path)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved.is_file():
            return str(resolved)

    return ref


def build_image_assets(
    *,
    asset_refs: list[str],
    citation_metadata: dict[str, Any] | None = None,
    document_name: str = "",
) -> list[dict[str, Any]]:
    citation_metadata = citation_metadata or {}
    assets: list[dict[str, Any]] = []
    for ref in asset_refs:
        path = Path(ref).expanduser()
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
            continue

        url = _artifact_image_url(path)
        if not url:
            continue

        assets.append(
            {
                "url": url,
                "filename": path.name,
                "page_num": citation_metadata.get("page_num"),
                "document_id": citation_metadata.get("document_id"),
                "document_name": document_name,
                "source_artifact_id": citation_metadata.get("source_artifact_id"),
                "source_artifact_type": citation_metadata.get("source_artifact_type"),
            }
        )
    return assets


def _artifact_image_url(path: Path) -> str | None:
    try:
        resolved = path.resolve()
        root = Path(settings.PARSED_ARTIFACTS_DIR).expanduser().resolve()
    except OSError:
        return None

    if root != resolved and root not in resolved.parents:
        return None

    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return None

    encoded = quote(relative.as_posix(), safe="/")
    return f"{settings.API_V1_STR}/artifacts/image?path={encoded}"


def _asset_signature(ref: str) -> str:
    path = Path(ref)
    if path.exists() and path.is_file():
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return f"sha256:{digest}"
        except OSError:
            return f"path:{path.expanduser().as_posix().lower()}"
    return f"path:{Path(ref).expanduser().as_posix().lower()}"
