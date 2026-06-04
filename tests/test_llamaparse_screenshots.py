"""
Tests for LlamaParse screenshot capture and asset_ref association.

Verifies that:
- render_page_screenshots() creates PNGs and returns {page_num: path}
- screenshot_refs() maps page numbers to paths correctly
- LlamaParseParser._build_page_chunks() populates asset_refs from screenshot_map
- LlamaParseParser.parse() wires document_id → screenshot rendering → asset_refs
- asset_refs=[] when document_id is None or fitz raises
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.parser.external.helper.screenshots import (
    render_page_screenshots,
    screenshot_refs,
)


# ---------------------------------------------------------------------------
# screenshot_refs
# ---------------------------------------------------------------------------


def test_screenshot_refs_returns_paths_for_matching_pages():
    smap = {1: "/artifacts/doc1/screenshots/page_1.png", 2: "/artifacts/doc1/screenshots/page_2.png"}
    result = screenshot_refs([1, 2], smap)
    assert result == ["/artifacts/doc1/screenshots/page_1.png", "/artifacts/doc1/screenshots/page_2.png"]


def test_screenshot_refs_skips_missing_pages():
    smap = {1: "/artifacts/doc1/screenshots/page_1.png"}
    result = screenshot_refs([1, 3], smap)
    assert result == ["/artifacts/doc1/screenshots/page_1.png"]


def test_screenshot_refs_returns_empty_for_none_map():
    assert screenshot_refs([1, 2], None) == []


def test_screenshot_refs_returns_empty_for_empty_map():
    assert screenshot_refs([1, 2], {}) == []


# ---------------------------------------------------------------------------
# render_page_screenshots — no document_id
# ---------------------------------------------------------------------------


def test_render_page_screenshots_returns_empty_when_no_document_id(tmp_path):
    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"fake")
    result = render_page_screenshots(dummy_pdf, None, [(1, "content")])
    assert result == {}


# ---------------------------------------------------------------------------
# render_page_screenshots — fitz failure
# ---------------------------------------------------------------------------


def test_render_page_screenshots_returns_empty_on_fitz_error(tmp_path, monkeypatch):
    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"fake")

    monkeypatch.setattr(
        "app.ingestion.parser.external.helper.screenshots.fitz.open",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("bad pdf")),
    )
    monkeypatch.setattr(
        "app.ingestion.parser.external.helper.screenshots.settings",
        MagicMock(PARSED_ARTIFACTS_DIR=str(tmp_path)),
    )

    result = render_page_screenshots(dummy_pdf, "doc-123", [(1, "text")])
    assert result == {}


# ---------------------------------------------------------------------------
# render_page_screenshots — success path
# ---------------------------------------------------------------------------


def test_render_page_screenshots_creates_png_per_page(tmp_path, monkeypatch):
    dummy_pdf = tmp_path / "sample.pdf"
    dummy_pdf.write_bytes(b"fake")

    # Build a fake fitz page/pixmap chain
    fake_pix = MagicMock()
    fake_pix.save = MagicMock(side_effect=lambda p: Path(p).write_bytes(b"\x89PNG"))

    fake_page = MagicMock()
    fake_page.get_pixmap = MagicMock(return_value=fake_pix)

    fake_doc = MagicMock()
    fake_doc.__len__ = MagicMock(return_value=2)
    fake_doc.__getitem__ = MagicMock(return_value=fake_page)

    monkeypatch.setattr(
        "app.ingestion.parser.external.helper.screenshots.fitz.open",
        lambda *a, **kw: fake_doc,
    )
    monkeypatch.setattr(
        "app.ingestion.parser.external.helper.screenshots.settings",
        MagicMock(PARSED_ARTIFACTS_DIR=str(tmp_path)),
    )

    page_sections = [(1, "page one text"), (2, "page two text")]
    result = render_page_screenshots(dummy_pdf, "doc-abc", page_sections)

    assert set(result.keys()) == {1, 2}
    for page_num, path_str in result.items():
        p = Path(path_str)
        assert p.name == f"page_{page_num}.png"
        assert p.parent.name == "screenshots"
        assert p.parent.parent.name == "doc-abc"


def test_render_page_screenshots_skips_pages_not_in_sections(tmp_path, monkeypatch):
    dummy_pdf = tmp_path / "sample.pdf"
    dummy_pdf.write_bytes(b"fake")

    fake_pix = MagicMock()
    fake_pix.save = MagicMock(side_effect=lambda p: Path(p).write_bytes(b"\x89PNG"))

    fake_page = MagicMock()
    fake_page.get_pixmap = MagicMock(return_value=fake_pix)

    fake_doc = MagicMock()
    fake_doc.__len__ = MagicMock(return_value=3)
    fake_doc.__getitem__ = MagicMock(return_value=fake_page)

    monkeypatch.setattr(
        "app.ingestion.parser.external.helper.screenshots.fitz.open",
        lambda *a, **kw: fake_doc,
    )
    monkeypatch.setattr(
        "app.ingestion.parser.external.helper.screenshots.settings",
        MagicMock(PARSED_ARTIFACTS_DIR=str(tmp_path)),
    )

    # Only page 2 has content
    result = render_page_screenshots(dummy_pdf, "doc-xyz", [(2, "only this page")])
    assert set(result.keys()) == {2}


# ---------------------------------------------------------------------------
# LlamaParseParser._build_page_chunks — asset_refs population
# ---------------------------------------------------------------------------


def test_llamaparse_build_page_chunks_populates_asset_refs():
    from app.ingestion.parser.external.llamacloud import LlamaParseParser

    parser = object.__new__(LlamaParseParser)
    from app.ingestion.parser.external.helper.md_metadata import MarkdownPageAnalyzer
    parser._analyzer = MarkdownPageAnalyzer(
        parser_name="llamaparse", layout_engine="llamaparse_vlm"
    )
    parser.PARSER_VERSION = "test"

    screenshot_map = {1: "/artifacts/doc1/screenshots/page_1.png"}
    page_sections = [(1, "Some text content on page one")]

    chunks = parser._build_page_chunks(page_sections, screenshot_map)

    assert len(chunks) == 1
    assert chunks[0].asset_refs == ["/artifacts/doc1/screenshots/page_1.png"]


def test_llamaparse_build_page_chunks_empty_asset_refs_when_no_map():
    from app.ingestion.parser.external.llamacloud import LlamaParseParser

    parser = object.__new__(LlamaParseParser)
    from app.ingestion.parser.external.helper.md_metadata import MarkdownPageAnalyzer
    parser._analyzer = MarkdownPageAnalyzer(
        parser_name="llamaparse", layout_engine="llamaparse_vlm"
    )
    parser.PARSER_VERSION = "test"

    chunks = parser._build_page_chunks([(1, "Some text")], screenshot_map=None)

    assert len(chunks) == 1
    assert chunks[0].asset_refs == []


# ---------------------------------------------------------------------------
# LlamaParseParser.parse() — document_id flows into screenshot rendering
# ---------------------------------------------------------------------------


def test_llamaparse_parse_calls_render_with_document_id(tmp_path, monkeypatch):
    """parse() must pass document_id to render_page_screenshots."""
    from app.ingestion.parser.external import llamacloud as llamacloud_mod

    captured: dict = {}

    def fake_render(pdf_path, document_id, page_sections):
        captured["document_id"] = document_id
        captured["page_sections"] = page_sections
        return {}

    monkeypatch.setattr(llamacloud_mod, "render_page_screenshots", fake_render)

    # Build minimal fake LlamaCloud client response
    fake_page = MagicMock()
    fake_page.markdown = "Some page text"
    fake_page.page_number = 1

    fake_markdown = MagicMock()
    fake_markdown.pages = [fake_page]

    fake_result = MagicMock()
    fake_result.markdown = fake_markdown
    fake_result.job = MagicMock(id="job-123")

    fake_files = MagicMock()
    fake_files.create = MagicMock(return_value=MagicMock(id="file-id-1"))

    fake_parsing = MagicMock()
    fake_parsing.parse = MagicMock(return_value=fake_result)

    fake_client = MagicMock()
    fake_client.files = fake_files
    fake_client.parsing = fake_parsing

    from app.ingestion.parser.external.llamacloud import LlamaParseParser
    from app.ingestion.parser.external.helper.md_metadata import MarkdownPageAnalyzer

    parser = object.__new__(LlamaParseParser)
    parser._client = fake_client
    parser._analyzer = MarkdownPageAnalyzer(
        parser_name="llamaparse", layout_engine="llamaparse_vlm"
    )
    parser.PARSER_NAME = "llamaparse"
    parser.LAYOUT_ENGINE = "llamaparse_vlm"
    parser.PARSER_VERSION = "test"

    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"fake")

    parsed = parser.parse(dummy_pdf, document_id="doc-999")

    assert captured.get("document_id") == "doc-999"
    assert len(parsed.page_chunks) == 1
    assert parsed.page_chunks[0].asset_refs == []  # fake_render returned {}
