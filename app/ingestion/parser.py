"""
Document parser facade.

- PDF: delegated to modular layout-aware pipeline.
- Non-PDF: legacy parsing path.
"""

import logging
import uuid
from pathlib import Path

from llama_index.core import Document as LlamaDocument
from llama_index.readers.file import DocxReader, PDFReader, PptxReader

from app.core.config import settings
from app.ingestion.pdf_pipeline.helpers import (
    has_chart_signals,
    has_numeric_data,
    has_table_signals,
)
from app.ingestion.pdf_pipeline.pipeline import parse_pdf_layout_aware

logger = logging.getLogger(__name__)
PARSER_NAME = "rag_parser"
PARSER_VERSION = settings.PDF_LAYOUT_PARSER_VERSION
LEGACY_PARSER_VERSION = "1.0.0"


def save_upload_file(file_content: bytes, filename: str, dest_folder: str) -> str:
    """Save uploaded file content to disk and return the path.

    Args:
        file_content: Raw bytes of the uploaded file.
        filename: Target filename to save as.
        dest_folder: Target directory to place the file in.

    Returns:
        Absolute string path where the file was saved.
    """
    dest_path = Path(dest_folder)
    dest_path.mkdir(parents=True, exist_ok=True)
    file_path = dest_path / filename
    file_path.write_bytes(file_content)
    return str(file_path)


def parse_document(
    file_path: str, document_metadata: dict
) -> tuple[list[LlamaDocument], list[dict]]:
    """Parse a document into LlamaIndex docs and per-unit signal metadata.

    Args:
        file_path: Path to the document.
        document_metadata: Extracted metadata tags.

    Returns:
        A tuple of (generated LlamaDocuments, extracted metadata units).
    """
    path = Path(file_path)

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
                "Layout-aware PDF parsing failed for %s. Falling back to legacy parser. "
                "Cause=%s: %s",
                path.name,
                type(exc).__name__,
                exc,
            )

    docs, units = _parse_legacy(file_path, document_metadata)
    logger.info(
        "Parsed %s via legacy path: %d documents, %d units",
        path.name,
        len(docs),
        len(units),
    )
    return docs, units


def _parse_legacy(
    file_path: str,
    document_metadata: dict,
) -> tuple[list[LlamaDocument], list[dict]]:
    """Parse via legacy extraction methods for non-PDF files or fallback.

    Args:
        file_path: Path to the document to parse.
        document_metadata: Metadata to insert into nodes.

    Returns:
        A tuple of (generated LlamaDocuments, metadata units).
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    docs: list[LlamaDocument] = []
    units: list[dict] = []

    def add_item(text: str, index: int, unit_type: str):
        page_or_slide = index + 1
        meta_key = "slide_num" if unit_type == "slide" else "page_num"

        meta = {
            **document_metadata,
            meta_key: page_or_slide,
            "source_file": path.name,
            "parser_name": PARSER_NAME,
            "parser_version": LEGACY_PARSER_VERSION,
        }
        docs.append(LlamaDocument(text=text, metadata=meta))

        units.append(
            {
                "id": str(uuid.uuid4()),
                "document_id": document_metadata["document_id"],
                "unit_type": unit_type,
                meta_key: page_or_slide,
                "raw_text": text,
                "table_detected": has_table_signals(text),
                "chart_detected": has_chart_signals(text),
                "contains_numeric_data": has_numeric_data(text),
            }
        )

    reader_map = {
        ".pdf": (PDFReader, "page"),
        ".pptx": (PptxReader, "slide"),
        ".docx": (DocxReader, "section"),
    }

    if ext in reader_map:
        reader_cls, unit_type = reader_map[ext]
        for i, doc in enumerate(reader_cls().load_data(file_path)):
            add_item(doc.text, i, unit_type)
    else:
        add_item(path.read_text(encoding="utf-8"), 0, "section")

    return docs, units


# Re-exported for tests and legacy imports.
_has_table_signals = has_table_signals
_has_chart_signals = has_chart_signals
_has_numeric_data = has_numeric_data
