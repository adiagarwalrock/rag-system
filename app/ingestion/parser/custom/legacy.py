import uuid
from pathlib import Path

from llama_index.core import Document as LlamaDocument
from llama_index.readers.file import DocxReader, PDFReader, PptxReader

from app.ingestion.parser.custom.pdf_pipeline.helpers import (
    has_chart_signals,
    has_numeric_data,
    has_table_signals,
)
from app.ingestion.retrieval_metadata import normalize_retrieval_metadata

PARSER_NAME = "rag_parser"
LEGACY_PARSER_VERSION = "1.0.0"


def run(
    file_path: str,
    document_metadata: dict,
) -> tuple[list[LlamaDocument], list[dict]]:
    """Parse via legacy extraction methods for non-PDF files or fallback."""
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
        meta = normalize_retrieval_metadata(
            meta,
            text=text,
            document_metadata=document_metadata,
            source_file=path.name,
        )
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
