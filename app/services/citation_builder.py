"""
Citation builder: constructs structured citations from retrieved source nodes.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List

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
            # Prefer the visible page number printed on the slide over the PDF page index
            "page_num": metadata.get("visible_page_num") or metadata.get("page_num"),
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
            "citation_label": metadata.get(
                "citation_label", _build_fallback_label(metadata, i + 1)
            ),
            "authority_score": metadata.get("authority_score"),
            "asset_refs": asset_refs,
            "has_image_assets": bool(asset_refs),
            "figure_type": metadata.get("figure_type"),
            "chart_type": metadata.get("chart_type"),
            "table_id": metadata.get("table_id"),
            "slide_purpose": metadata.get("slide_purpose"),
            "image_scope": metadata.get("image_scope"),
            "chart_parse_status": metadata.get("chart_parse_status"),
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

    display_page = metadata.get("visible_page_num") or metadata.get("page_num")
    if display_page:
        parts.append(f"p.{display_page}")
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


def _asset_signature(ref: str) -> str:
    path = Path(ref)
    if path.exists() and path.is_file():
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return f"sha256:{digest}"
        except OSError:
            return f"path:{path.expanduser().as_posix().lower()}"
    return f"path:{Path(ref).expanduser().as_posix().lower()}"
