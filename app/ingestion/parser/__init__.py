"""
Document parser package.

Public API:
  parse_document(file_path, document_metadata) -> (list[LlamaDocument], list[dict])
  save_upload_file(file_content, filename, dest_folder) -> str

Routing (priority order, each falls through on failure or missing key):
  1. Reducto          — if ENABLE_EXTERNAL_PARSER and REDUCTO_API_KEY is set
  2. LlamaParse       — if ENABLE_EXTERNAL_PARSER and LLAMAPARSE_API_KEY is set
  3. Layout-aware PDF — if PDF and ENABLE_LAYOUT_AWARE_PDF
  4. Legacy           — always available (LlamaIndex readers)
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from llama_index.core import Document as LlamaDocument

from app.core.config import settings
from app.ingestion.parser.custom import parse_pdf_layout_aware, run_legacy
from app.ingestion.parser.custom.pdf_pipeline.helpers import (
    has_chart_signals,
    has_numeric_data,
    has_table_signals,
)

logger = logging.getLogger(__name__)

PARSER_NAME = "rag_parser"
PARSER_VERSION = settings.PDF_LAYOUT_PARSER_VERSION
LEGACY_PARSER_VERSION = "1.0.0"

_PAGE_START = re.compile(r"^\[\[START OF PAGE (\d+)\]\]$")
_PAGE_END = re.compile(r"^\[\[END OF PAGE (\d+)\]\]$")


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------


def save_upload_file(file_content: bytes, filename: str, dest_folder: str) -> str:
    """Save uploaded file content to disk and return the path."""
    dest_path = Path(dest_folder)
    dest_path.mkdir(parents=True, exist_ok=True)
    file_path = dest_path / filename
    file_path.write_bytes(file_content)
    return str(file_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_document(
    file_path: str, document_metadata: dict
) -> tuple[list[LlamaDocument], list[dict]]:
    """Parse a document into LlamaIndex docs and per-unit signal metadata."""
    path = Path(file_path)

    # Level 1 — Reducto
    if settings.ENABLE_EXTERNAL_PARSER and settings.REDUCTO_API_KEY:
        try:
            from app.ingestion.parser.external.reducto import run

            md, elapsed = run(path)
            docs, units = to_llama_docs(md, document_metadata, parser_name="reducto")
            logger.info(
                "Parsed %s via reducto in %.2fs: %d chunks",
                path.name,
                elapsed,
                len(docs),
            )
            return docs, units
        except Exception as exc:
            logger.warning(
                "Reducto failed for %s. Trying LlamaParse. Cause=%s: %s",
                path.name,
                type(exc).__name__,
                exc,
            )

    # Level 2 — LlamaParse
    if settings.ENABLE_EXTERNAL_PARSER and settings.LLAMAPARSE_API_KEY:
        try:
            from app.ingestion.parser.external.llamacloud import run

            md, elapsed = run(path)
            docs, units = to_llama_docs(md, document_metadata, parser_name="llamaparse")
            logger.info(
                "Parsed %s via llamaparse in %.2fs: %d chunks",
                path.name,
                elapsed,
                len(docs),
            )
            return docs, units
        except Exception as exc:
            logger.warning(
                "LlamaParse failed for %s. Trying layout-aware. Cause=%s: %s",
                path.name,
                type(exc).__name__,
                exc,
            )

    # Level 3 — Layout-aware PDF
    if path.suffix.lower() == ".pdf" and settings.ENABLE_LAYOUT_AWARE_PDF:
        try:
            return parse_pdf_layout_aware(file_path, document_metadata)
        except Exception as exc:
            if settings.STRICT_LAYOUT_AWARE_PDF_FAILURE:
                logger.exception(
                    "Layout-aware PDF parsing failed for %s with strict mode enabled.",
                    path.name,
                )
                raise
            logger.warning(
                "Layout-aware PDF parsing failed for %s. Falling back to legacy. Cause=%s: %s",
                path.name,
                type(exc).__name__,
                exc,
            )

    # Level 4 — Legacy (always available)
    docs, units = run_legacy(file_path, document_metadata)
    logger.info(
        "Parsed %s via legacy: %d documents, %d units", path.name, len(docs), len(units)
    )
    return docs, units


# ---------------------------------------------------------------------------
# Page-marker parsing
# ---------------------------------------------------------------------------


def _split_by_page_markers(markdown: str) -> list[tuple[int, str]]:
    """Split markdown on [[START OF PAGE n]] / [[END OF PAGE n]] pairs.

    Returns [(page_num, content), ...]. Falls back to [(1, markdown)] when no
    markers are present so callers always get at least one page.
    """
    pages: list[tuple[int, str]] = []
    current_page: int | None = None
    buffer: list[str] = []

    for line in markdown.splitlines():
        m_start = _PAGE_START.match(line.strip())
        m_end = _PAGE_END.match(line.strip())
        if m_start:
            buffer = []
            current_page = int(m_start.group(1))
        elif m_end:
            if current_page is not None:
                pages.append((current_page, "\n".join(buffer).strip()))
            buffer = []
            current_page = None
        else:
            buffer.append(line)

    if not pages:
        # No markers — treat whole document as page 1
        return [(1, markdown.strip())]

    return [(p, c) for p, c in pages if c]


# ---------------------------------------------------------------------------
# Markdown → LlamaDocument converter (used by external parsers)
# ---------------------------------------------------------------------------


def to_llama_docs(
    markdown: str,
    document_metadata: dict,
    parser_name: str,
    parser_version: str = "1.0.0",
) -> tuple[list[LlamaDocument], list[dict]]:
    """Convert page-delimited markdown into (docs, units) matching run_legacy() shape."""
    pages = _split_by_page_markers(markdown)

    docs: list[LlamaDocument] = []
    units: list[dict] = []

    for page_num, page_content in pages:
        meta = {
            **document_metadata,
            "page_num": page_num,
            "parser_name": parser_name,
            "parser_version": parser_version,
        }
        docs.append(LlamaDocument(text=page_content, metadata=meta))
        units.append(
            {
                "id": str(uuid.uuid4()),
                "document_id": document_metadata["document_id"],
                "unit_type": "page",
                "page_num": page_num,
                "raw_text": page_content,
                "table_detected": has_table_signals(page_content),
                "chart_detected": has_chart_signals(page_content),
                "contains_numeric_data": has_numeric_data(page_content),
            }
        )

    return docs, units
