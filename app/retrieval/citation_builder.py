"""
Citation builder: constructs structured citations from retrieved source nodes.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


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

    for i, node in enumerate(source_nodes):
        metadata = node.node.metadata or {}
        text = node.node.text or ""

        citation = {
            "rank": i + 1,
            "score": round(node.score, 4) if node.score else None,
            "text": text[:500],
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
            "chunk_type": metadata.get("chunk_type", "text"),
            "version_label": metadata.get("version_label"),
            "version_group": metadata.get("document_version_group"),
            "effective_from": metadata.get("effective_from"),
            "effective_to": metadata.get("effective_to"),
            "citation_label": metadata.get(
                "citation_label", _build_fallback_label(metadata, i + 1)
            ),
            "authority_score": metadata.get("authority_score"),
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
        text_preview = c["text"][:300].replace("\n", " ")
        lines.append(f"[{label}]: {text_preview}")
    return "\n\n".join(lines)
