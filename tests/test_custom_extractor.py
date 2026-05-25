from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pymupdf as fitz
import pytest

from app.ingestion.parser.external.helper.chunk_schema import (
    ProviderMetadataAssociationExtraction,
)
from app.scripts import custom_extractor


def test_parse_page_ranges_accepts_pages_ranges_and_dedupes():
    pages = custom_extractor.parse_page_ranges("1,3,3,5-7", total_pages=10)

    assert pages == [1, 3, 5, 6, 7]


def test_parse_page_ranges_defaults_to_all_or_max_pages():
    assert custom_extractor.parse_page_ranges(None, total_pages=4) == [1, 2, 3, 4]
    assert custom_extractor.parse_page_ranges(None, total_pages=4, max_pages=2) == [
        1,
        2,
    ]


def test_parse_page_ranges_rejects_invalid_selection():
    with pytest.raises(ValueError, match="outside the PDF page count"):
        custom_extractor.parse_page_ranges("1,5", total_pages=2)

    with pytest.raises(ValueError, match="exceeds --max-pages"):
        custom_extractor.parse_page_ranges("1-3", total_pages=5, max_pages=2)


def test_render_pdf_pages_writes_selected_images(tmp_path):
    pdf_path = tmp_path / "Financial Deck.pdf"
    _create_sample_pdf(pdf_path, page_count=3)

    rendered = custom_extractor.render_pdf_pages(
        pdf_path,
        pages=[1, 3],
        output_dir=tmp_path / "out",
        dpi=72,
    )

    assert [page.page_num for page in rendered] == [1, 3]
    assert [page.chunk_id for page in rendered] == [
        "Financial_Deck-p1",
        "Financial_Deck-p3",
    ]
    assert all(page.image_path.exists() for page in rendered)


def test_validate_extraction_shape_accepts_matching_chunks(tmp_path):
    rendered = [
        custom_extractor.RenderedPage(
            page_num=1,
            chunk_id="sample-p1",
            image_path=tmp_path / "page_0001.png",
        )
    ]
    extraction = _sample_extraction()

    custom_extractor.validate_extraction_shape(extraction, rendered)


def test_validate_extraction_shape_rejects_missing_chunks(tmp_path):
    rendered = [
        custom_extractor.RenderedPage(
            page_num=2,
            chunk_id="sample-p2",
            image_path=tmp_path / "page_0002.png",
        )
    ]

    with pytest.raises(ValueError, match="unexpected chunk IDs"):
        custom_extractor.validate_extraction_shape(_sample_extraction(), rendered)


def test_module_import_has_no_cli_side_effects():
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("AI_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import app.scripts.custom_extractor; print('imported')",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


def test_write_outputs_creates_json_and_markdown(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    rendered = [
        custom_extractor.RenderedPage(
            page_num=1,
            chunk_id="sample-p1",
            image_path=tmp_path / "pages" / "page_0001.png",
        )
    ]
    extraction = _sample_extraction()

    artifacts = custom_extractor.write_outputs(
        pdf_path=pdf_path,
        output_dir=tmp_path / "out",
        model="gpt-test",
        rendered_pages=rendered,
        extraction=extraction,
        timings={"render_seconds": 0.1, "extract_seconds": 0.2, "total_seconds": 0.3},
    )

    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert payload["validation"]["status"] == "passed"
    assert payload["selected_pages"] == [1]
    assert payload["extraction"]["chunks"][0]["chunk_id"] == "sample-p1"
    assert "# Custom Extractor Review: sample.pdf" in markdown
    assert "Revenue table" in markdown
    assert "NOI was 5" in markdown


def test_cli_fails_before_api_call_when_api_key_missing(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    _create_sample_pdf(pdf_path, page_count=1)
    monkeypatch.setattr(custom_extractor.settings, "AI_API_KEY", None)

    args = Namespace(
        pdf_path=str(pdf_path),
        pages="1",
        output_dir=str(tmp_path / "out"),
        model="gpt-test",
        dpi=72,
        max_pages=None,
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY or AI_API_KEY"):
        custom_extractor.run_extraction_cli(args)


def _sample_extraction() -> ProviderMetadataAssociationExtraction:
    return ProviderMetadataAssociationExtraction.model_validate(
        {
            "document_metadata": {
                "document_id": "sample",
                "file_name": "sample.pdf",
                "parser_name": "custom_extractor",
                "parser_version": "0.1.0",
            },
            "chunks": [
                {
                    "chunk_id": "sample-p1",
                    "page_nums": [1],
                    "metadata": {
                        "document_date": "2026-05",
                        "table_title": "Revenue table",
                        "key_chart_facts": ["NOI was 5"],
                        "llm_enrichment_confidence": 0.8,
                    },
                    "citations": [
                        {
                            "text": "Revenue table",
                            "page_num": 1,
                            "confidence": 0.9,
                        }
                    ],
                    "extraction_confidence": 0.8,
                }
            ],
        }
    )


def _create_sample_pdf(path: Path, *, page_count: int) -> None:
    doc = fitz.open()
    for page_index in range(page_count):
        page = doc.new_page(width=300, height=200)
        page.insert_text((36, 60), f"Page {page_index + 1}")
        page.insert_text((36, 90), "Revenue | NOI | Occupancy")
    doc.save(str(path))
    doc.close()
