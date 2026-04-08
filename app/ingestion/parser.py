"""
Enhanced document parser with page-level extraction and metadata injection.
"""

import logging
import os
import shutil
import uuid
from typing import List, Tuple

from llama_index.core import Document as LlamaDocument
from llama_index.readers.file import DocxReader, PDFReader, PptxReader

logger = logging.getLogger(__name__)
PARSER_NAME = "rag_parser"
PARSER_VERSION = "1.0.0"


def save_upload_file(file_content: bytes, filename: str, dest_folder: str) -> str:
    """Save uploaded file content to disk and return the path."""
    os.makedirs(dest_folder, exist_ok=True)
    file_path = os.path.join(dest_folder, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)
    return file_path


def parse_document(
    file_path: str, document_metadata: dict
) -> Tuple[List[LlamaDocument], List[dict]]:
    """Parse a document into LlamaIndex docs and per-unit signal metadata."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    filename = os.path.basename(file_path)

    docs: List[LlamaDocument] = []
    units: List[dict] = []

    def add_item(text: str, index: int, unit_type: str):
        page_or_slide = index + 1
        meta_key = "slide_num" if unit_type == "slide" else "page_num"

        # Create LlamaIndex Document
        meta = {
            **document_metadata,
            meta_key: page_or_slide,
            "source_file": filename,
            "parser_name": PARSER_NAME,
            "parser_version": PARSER_VERSION,
        }
        docs.append(LlamaDocument(text=text, metadata=meta))

        # Create parsed unit metadata used for ingestion-time signals
        units.append(
            {
                "id": str(uuid.uuid4()),
                "document_id": document_metadata["document_id"],
                "unit_type": unit_type,
                meta_key: page_or_slide,
                "raw_text": text,
                "table_detected": _has_table_signals(text),
                "chart_detected": _has_chart_signals(text),
                "contains_numeric_data": _has_numeric_data(text),
            }
        )

    if ext == ".pdf":
        raw_docs = PDFReader().load_data(file_path)
        for i, doc in enumerate(raw_docs):
            add_item(doc.text, i, "page")

    elif ext == ".pptx":
        raw_docs = PptxReader().load_data(file_path)
        for i, doc in enumerate(raw_docs):
            add_item(doc.text, i, "slide")

    elif ext == ".docx":
        raw_docs = DocxReader().load_data(file_path)
        for i, doc in enumerate(raw_docs):
            add_item(doc.text, i, "section")

    else:
        with open(file_path, "r") as f:
            text = f.read()
        add_item(text, 0, "section")

    logger.info("Parsed %s: %d documents, %d units", filename, len(docs), len(units))
    return docs, units


def _has_table_signals(text: str) -> bool:
    """Heuristic: detect table-like structures in text."""
    if not text:
        return False
    # Detect tabs or pipe characters in multiple lines
    lines = text.strip().split("\n")
    return sum(1 for l in lines if "\t" in l or l.count("|") >= 2) >= 2


def _has_chart_signals(text: str) -> bool:
    """Heuristic: detect chart/figure references."""
    keywords = ["figure", "chart", "graph", "plot", "diagram"]
    return any(kw in (text or "").lower() for kw in keywords)


def _has_numeric_data(text: str) -> bool:
    """Heuristic: detect significant numeric content."""
    import re

    return len(re.findall(r"\b\d+[.,]?\d*%?\b", text or "")) >= 3
