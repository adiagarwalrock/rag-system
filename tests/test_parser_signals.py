from pathlib import Path

import pymupdf as fitz
import pytest

from app.ingestion.parser import (
    _has_chart_signals,
    _has_numeric_data,
    _has_table_signals,
    parse_document,
)


def test_has_table_signals_detects_multi_line_table_patterns():
    text = "col1|col2|col3\n1|2|3\nplain text"
    assert _has_table_signals(text) is True


def test_has_table_signals_ignores_single_line_delimiters():
    text = "just one line with a|b|c"
    assert _has_table_signals(text) is False


def test_has_chart_signals_detects_common_chart_keywords():
    assert _has_chart_signals("Figure 3 shows quarterly growth.") is True
    assert _has_chart_signals("No visual references here") is False


def test_has_numeric_data_requires_at_least_three_numbers():
    assert _has_numeric_data("values: 10, 20, and 30") is True
    assert _has_numeric_data("values: 10 and 20") is False


def test_parse_document_fallback_text_path_emits_unit_metadata(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "Revenue was 10 in Q1, 20 in Q2, and 30 in Q3.", encoding="utf-8"
    )

    metadata = {
        "document_id": "doc-123",
        "client_id": "client-1",
        "client_name": "Acme",
    }

    docs, units = parse_document(str(file_path), metadata)

    assert len(docs) == 1
    assert len(units) == 1
    assert docs[0].metadata["page_num"] == 1
    assert docs[0].metadata["source_file"] == "sample.txt"
    assert units[0]["document_id"] == "doc-123"
    assert units[0]["unit_type"] == "section"
    assert units[0]["page_num"] == 1
    assert units[0]["contains_numeric_data"] is True


def test_parse_document_pdf_layout_path_emits_artifact_chunks(tmp_path):
    file_path = tmp_path / "financial.pdf"
    _create_sample_financial_pdf(file_path)

    metadata = {
        "document_id": "doc-pdf-1",
        "client_id": "client-1",
        "client_name": "Acme",
    }
    docs, units = parse_document(str(file_path), metadata)

    assert docs
    assert units
    chunk_types = {doc.metadata.get("chunk_type") for doc in docs}
    assert any(kind in chunk_types for kind in {"full_table", "table_segment"})
    assert any(kind in chunk_types for kind in {"figure_artifact", "visual_proxy_text"})
    assert "chart_data_points" in chunk_types
    assert "page_card" in chunk_types

    artifact_bundle_path = docs[0].metadata.get("artifact_bundle_path")
    assert artifact_bundle_path
    assert (Path(artifact_bundle_path) / "chunk_artifacts.json").exists()


def test_parse_document_pdf_layout_strict_mode_raises(monkeypatch, tmp_path):
    file_path = tmp_path / "strict-financial.pdf"
    _create_sample_financial_pdf(file_path)

    metadata = {
        "document_id": "doc-pdf-strict",
        "client_id": "client-1",
        "client_name": "Acme",
    }

    def _raise_layout_error(*_args, **_kwargs):
        raise RuntimeError("layout parsing failed")

    monkeypatch.setattr(
        "app.ingestion.parser.parse_pdf_layout_aware", _raise_layout_error
    )
    monkeypatch.setattr("app.ingestion.parser.settings.ENABLE_LAYOUT_AWARE_PDF", True)
    monkeypatch.setattr(
        "app.ingestion.parser.settings.STRICT_LAYOUT_AWARE_PDF_FAILURE", True
    )

    with pytest.raises(RuntimeError, match="layout parsing failed"):
        parse_document(str(file_path), metadata)


def test_parse_document_pdf_layout_non_strict_falls_back(monkeypatch, tmp_path):
    file_path = tmp_path / "nonstrict-financial.pdf"
    _create_sample_financial_pdf(file_path)

    metadata = {
        "document_id": "doc-pdf-nonstrict",
        "client_id": "client-1",
        "client_name": "Acme",
    }

    def _raise_layout_error(*_args, **_kwargs):
        raise RuntimeError("layout parsing failed")

    monkeypatch.setattr(
        "app.ingestion.parser.parse_pdf_layout_aware", _raise_layout_error
    )
    monkeypatch.setattr("app.ingestion.parser.settings.ENABLE_LAYOUT_AWARE_PDF", True)
    monkeypatch.setattr(
        "app.ingestion.parser.settings.STRICT_LAYOUT_AWARE_PDF_FAILURE", False
    )

    docs, units = parse_document(str(file_path), metadata)
    assert docs
    assert units
    assert docs[0].metadata["parser_version"] == "1.0.0"


def _create_sample_financial_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "Financial Summary")
    page.insert_text((72, 95), "Table 1 Revenue by Quarter")
    page.insert_text((72, 120), "Quarter | Revenue | Growth")
    page.insert_text((72, 140), "Q1 | 10 | 5%")
    page.insert_text((72, 160), "Q2 | 20 | 8%")
    page.insert_text((72, 180), "Q3 | 30 | 12%")
    page.insert_text((72, 235), "Figure 1 Revenue Trend")
    for idx in range(30):
        top = 255 + (idx * 3)
        page.draw_line((80, top), (300, top + 1))
    page.insert_text(
        (72, 420),
        "The chart highlights sustained growth from 10 to 30 over the first three quarters.",
    )
    doc.save(str(path))
    doc.close()
