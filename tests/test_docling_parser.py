"""Tests for the Docling internal parser."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers for building synthetic Docling objects
# ---------------------------------------------------------------------------


def _prov(page_no: int):
    return SimpleNamespace(page_no=page_no)


def _text_item(text: str, page_no: int = 1):
    from docling_core.types.doc.document import TextItem

    item = MagicMock(spec=TextItem)
    item.text = text
    item.prov = [_prov(page_no)]
    return item


def _section_header(text: str, page_no: int = 1):
    from docling_core.types.doc.document import SectionHeaderItem

    item = MagicMock(spec=SectionHeaderItem)
    item.text = text
    item.prov = [_prov(page_no)]
    return item


def _table_item(markdown: str, page_no: int = 1):
    from docling_core.types.doc.document import TableItem

    item = MagicMock(spec=TableItem)
    item.export_to_markdown.return_value = markdown
    item.prov = [_prov(page_no)]
    return item


def _picture_item(caption: str, page_no: int = 1):
    from docling_core.types.doc.document import PictureItem

    item = MagicMock(spec=PictureItem)
    item.caption_text = MagicMock(return_value=caption)
    item.prov = [_prov(page_no)]
    return item


def _make_doc(items: list):
    """Build a minimal fake DoclingDocument whose iterate_items() yields given items."""
    doc = MagicMock()
    doc.pages = {1: MagicMock(), 2: MagicMock()}
    doc.iterate_items.return_value = [(item, None) for item in items]
    return doc


# ---------------------------------------------------------------------------
# iter_blocks tests
# ---------------------------------------------------------------------------


class TestIterBlocks:
    def test_text_item_yields_text_block(self):
        from app.ingestion.parser.custom.docling_parser import iter_blocks

        doc = _make_doc([_text_item("Hello world", page_no=2)])
        blocks = list(iter_blocks(doc))

        assert len(blocks) == 1
        assert blocks[0].text == "Hello world"
        assert blocks[0].content_type == "text"
        assert blocks[0].page_number == 2

    def test_section_header_updates_section_and_emits_text(self):
        from app.ingestion.parser.custom.docling_parser import iter_blocks

        doc = _make_doc([
            _section_header("Revenue Overview", page_no=1),
            _text_item("Q4 was strong.", page_no=1),
        ])
        blocks = list(iter_blocks(doc))

        assert len(blocks) == 2
        assert blocks[0].content_type == "text"
        assert blocks[0].section_title == "Revenue Overview"
        assert blocks[1].section_title == "Revenue Overview"

    def test_table_item_yields_table_block(self):
        from app.ingestion.parser.custom.docling_parser import iter_blocks

        doc = _make_doc([_table_item("| A | B |\n|---|---|\n| 1 | 2 |", page_no=3)])
        blocks = list(iter_blocks(doc))

        assert len(blocks) == 1
        assert blocks[0].content_type == "table"
        assert "A" in blocks[0].text

    def test_picture_with_caption_yields_chart_caption(self):
        from app.ingestion.parser.custom.docling_parser import iter_blocks

        doc = _make_doc([_picture_item("Figure 1: Revenue chart", page_no=1)])
        blocks = list(iter_blocks(doc))

        assert len(blocks) == 1
        assert blocks[0].content_type == "chart_caption"
        assert "Revenue chart" in blocks[0].text

    def test_picture_without_caption_is_skipped(self):
        from app.ingestion.parser.custom.docling_parser import iter_blocks

        doc = _make_doc([_picture_item("", page_no=1)])
        blocks = list(iter_blocks(doc))

        assert len(blocks) == 0

    def test_empty_text_items_are_skipped(self):
        from app.ingestion.parser.custom.docling_parser import iter_blocks

        doc = _make_doc([_text_item("   ", page_no=1)])
        blocks = list(iter_blocks(doc))

        assert len(blocks) == 0


# ---------------------------------------------------------------------------
# run() contract tests
# ---------------------------------------------------------------------------


_DOC_META = {
    "document_id": "doc-abc",
    "client_id": "client-1",
    "file_name": "test.pdf",
}


class TestRun:
    def _make_fake_doc(self):
        return _make_doc([
            _section_header("Overview", page_no=1),
            _text_item("Net revenue grew 12%.", page_no=1),
            _table_item("| Year | Revenue |\n|------|---------|", page_no=2),
            _picture_item("Chart: Revenue trend", page_no=2),
        ])

    def test_returns_docs_and_units(self, tmp_path):
        from app.ingestion.parser.custom.docling_parser import run

        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")

        fake_doc = self._make_fake_doc()
        with patch(
            "app.ingestion.parser.custom.docling_parser.parse_pdf",
            return_value=fake_doc,
        ):
            docs, units = run(fake_pdf, _DOC_META)

        assert len(docs) > 0
        assert len(units) == len(docs)

    def test_docs_have_correct_parser_name(self, tmp_path):
        from app.ingestion.parser.custom.docling_parser import PARSER_NAME, run

        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")

        with patch(
            "app.ingestion.parser.custom.docling_parser.parse_pdf",
            return_value=self._make_fake_doc(),
        ):
            docs, _ = run(fake_pdf, _DOC_META)

        for doc in docs:
            assert doc.metadata.get("parser_name") == PARSER_NAME

    def test_chunk_type_values_are_valid(self, tmp_path):
        from app.ingestion.parser.custom.docling_parser import run

        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")

        with patch(
            "app.ingestion.parser.custom.docling_parser.parse_pdf",
            return_value=self._make_fake_doc(),
        ):
            docs, _ = run(fake_pdf, _DOC_META)

        valid_types = {"text", "table", "chart_caption"}
        for doc in docs:
            assert doc.metadata.get("chunk_type") in valid_types

    def test_units_have_document_id(self, tmp_path):
        from app.ingestion.parser.custom.docling_parser import run

        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")

        with patch(
            "app.ingestion.parser.custom.docling_parser.parse_pdf",
            return_value=self._make_fake_doc(),
        ):
            _, units = run(fake_pdf, _DOC_META)

        for unit in units:
            assert unit["document_id"] == "doc-abc"

    def test_chart_caption_sets_chart_detected(self, tmp_path):
        from app.ingestion.parser.custom.docling_parser import run

        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")

        doc = _make_doc([_picture_item("Revenue chart caption", page_no=1)])
        with patch(
            "app.ingestion.parser.custom.docling_parser.parse_pdf",
            return_value=doc,
        ):
            _, units = run(fake_pdf, _DOC_META)

        assert len(units) == 1
        assert units[0]["chart_detected"] is True


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_docling_in_get_available_parsers(self):
        from app.ingestion.parser.registry import get_available_parsers

        parsers = get_available_parsers()
        ids = [p["id"] for p in parsers]
        assert "docling" in ids

    def test_docling_available_when_enabled(self):
        from app.ingestion.parser.registry import is_available

        cfg = MagicMock()
        cfg.ENABLE_DOCLING_PARSER = True
        assert is_available("docling", settings_obj=cfg) is True

    def test_docling_unavailable_when_disabled(self):
        from app.ingestion.parser.registry import is_available

        cfg = MagicMock()
        cfg.ENABLE_DOCLING_PARSER = False
        assert is_available("docling", settings_obj=cfg) is False

    def test_docling_in_valid_parsers(self):
        from app.ingestion.parser.registry import VALID_PARSERS

        assert "docling" in VALID_PARSERS
