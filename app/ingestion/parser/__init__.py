"""
Document parser package.

Public API:
  parse_document(file_path, document_metadata) -> (list[LlamaDocument], list[dict])
  save_upload_file(file_content, filename, dest_folder) -> str

Routing (priority order, each falls through on failure or missing key):
  1. Reducto          — if ENABLE_EXTERNAL_PARSER and REDUCTO_API_KEY is set
  2. LlamaParse       — if ENABLE_EXTERNAL_PARSER and LLAMAPARSE_API_KEY is set
  3. Layout-aware PDF — if PDF and ENABLE_LAYOUT_AWARE_PDF
  3b. Docling         — if PDF and ENABLE_DOCLING_PARSER (local, no API key)
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
from app.ingestion.parser.registry import (
    PARSER_AUTO,
    PARSER_DOCLING,
    PARSER_LAYOUT,
    PARSER_LEGACY,
    PARSER_LLAMA,
    PARSER_REDUCTO,
    is_available,
)
from app.ingestion.retrieval_metadata import normalize_retrieval_metadata

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
    file_path: str,
    document_metadata: dict,
    parser_preference: str | None = None,
) -> tuple[list[LlamaDocument], list[dict]]:
    """Parse a document into LlamaIndex docs and per-unit signal metadata.

    When parser_preference is None or "auto", the 5-level fallback chain runs.
    When a specific parser is named, it is invoked directly with no fallback.
    """

    path = Path(file_path)

    preference = (parser_preference or PARSER_AUTO).lower()

    if preference != PARSER_AUTO:
        return _parse_with_specific_parser(preference, path, file_path, document_metadata)

    # Auto fallback chain (levels 1-4)

    # Level 1 — Reducto
    if is_available(PARSER_REDUCTO):
        try:
            from app.ingestion.parser.external.reducto import run

            docs, units = run(path, document_metadata)
            logger.info("Parsed %s via reducto: %d chunks", path.name, len(docs))
            return docs, units
        except Exception as exc:
            logger.warning(
                "Reducto failed for %s. Trying LlamaParse. Cause=%s: %s",
                path.name, type(exc).__name__, exc,
            )

    # Level 2 — LlamaParse
    if is_available(PARSER_LLAMA):
        try:
            from app.ingestion.parser.external.llamacloud import run

            docs, units = run(path, document_metadata)
            logger.info("Parsed %s via llamaparse: %d chunks", path.name, len(docs))
            return docs, units
        except Exception as exc:
            logger.warning(
                "LlamaParse failed for %s. Trying layout-aware. Cause=%s: %s",
                path.name, type(exc).__name__, exc,
            )

    # Level 3 — Layout-aware PDF
    if path.suffix.lower() == ".pdf" and is_available(PARSER_LAYOUT):
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
                path.name, type(exc).__name__, exc,
            )

    # Level 3b — Docling (local, PDF only)
    if path.suffix.lower() == ".pdf" and is_available(PARSER_DOCLING):
        try:
            from app.ingestion.parser.custom.docling_parser import run as run_docling

            docs, units = run_docling(file_path, document_metadata)
            logger.info("Parsed %s via docling: %d chunks", path.name, len(docs))
            return docs, units
        except Exception as exc:
            logger.warning(
                "Docling failed for %s. Falling back to legacy. Cause=%s: %s",
                path.name, type(exc).__name__, exc,
            )

    # Level 4 — Legacy (always available)
    docs, units = run_legacy(file_path, document_metadata)
    logger.info(
        "Parsed %s via legacy: %d documents, %d units", path.name, len(docs), len(units)
    )
    return docs, units


def _parse_with_specific_parser(
    parser: str,
    path: Path,
    file_path: str,
    document_metadata: dict,
) -> tuple[list[LlamaDocument], list[dict]]:
    """Invoke a named parser directly, raising ValueError if unavailable."""
    if parser == PARSER_REDUCTO:
        if not is_available(PARSER_REDUCTO):
            raise ValueError(
                "Reducto parser is not available. Set ENABLE_EXTERNAL_PARSER=true and REDUCTO_API_KEY."
            )
        from app.ingestion.parser.external.reducto import run

        docs, units = run(path, document_metadata)
        logger.info("Parsed %s via reducto (explicit): %d chunks", path.name, len(docs))
        return docs, units

    if parser == PARSER_LLAMA:
        if not is_available(PARSER_LLAMA):
            raise ValueError(
                "LlamaParse is not available. Set ENABLE_EXTERNAL_PARSER=true and LLAMA_CLOUD_API_KEY."
            )
        from app.ingestion.parser.external.llamacloud import run

        docs, units = run(path, document_metadata)
        logger.info("Parsed %s via llamaparse (explicit): %d chunks", path.name, len(docs))
        return docs, units

    if parser == PARSER_LAYOUT:
        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Layout-aware parser only supports PDF files, got '{path.suffix}'."
            )
        if not is_available(PARSER_LAYOUT):
            raise ValueError(
                "Layout-aware PDF parser is disabled. Set ENABLE_LAYOUT_AWARE_PDF=true."
            )
        docs, units = parse_pdf_layout_aware(file_path, document_metadata)
        logger.info("Parsed %s via layout-aware (explicit): %d docs", path.name, len(docs))
        return docs, units

    if parser == PARSER_DOCLING:
        if not is_available(PARSER_DOCLING):
            raise ValueError("Docling parser is disabled. Set ENABLE_DOCLING_PARSER=true.")
        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Docling parser only supports PDF files, got '{path.suffix}'."
            )
        from app.ingestion.parser.custom.docling_parser import run as run_docling

        docs, units = run_docling(file_path, document_metadata)
        logger.info("Parsed %s via docling (explicit): %d chunks", path.name, len(docs))
        return docs, units

    # PARSER_LEGACY — always available, no guard needed
    docs, units = run_legacy(file_path, document_metadata)
    logger.info(
        "Parsed %s via legacy (explicit): %d documents, %d units",
        path.name, len(docs), len(units),
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
        meta = normalize_retrieval_metadata(
            meta,
            text=page_content,
            document_metadata=document_metadata,
            source_file=document_metadata.get("file_name")
            or document_metadata.get("document_name"),
        )
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
