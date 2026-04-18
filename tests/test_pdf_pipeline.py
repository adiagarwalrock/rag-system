from pathlib import Path

from app.core.config import settings
from app.ingestion.pdf_pipeline import artifact_builders
from app.ingestion.pdf_pipeline import adapters, page_structure, repair
from app.ingestion.pdf_pipeline.models import FigureArtifact, PageManifest


class _FakeTextItem:
    def __init__(self):
        self.text = "Revenue"
        self.x = 10
        self.y = 20
        self.width = 30
        self.height = 12


class _FakePage:
    def __init__(self):
        self.pageNum = 1
        self.width = 600
        self.height = 800
        self.text = "Revenue increased"
        self.textItems = [_FakeTextItem()]


class _FakeParseResult:
    def __init__(self):
        self.pages = [_FakePage()]


class _FakeScreenshot:
    def __init__(self):
        self.page_num = 1
        self.image_path = "/tmp/page_1.png"


class _FakeScreenshotBatch:
    def __init__(self):
        self.screenshots = [_FakeScreenshot()]


class _FakeLiteParse:
    def __init__(self, install_if_not_available: bool = False):
        self.install_if_not_available = install_if_not_available

    def parse(self, *_args, **_kwargs):
        return _FakeParseResult()

    def screenshot(self, *_args, **_kwargs):
        return _FakeScreenshotBatch()


def test_extract_liteparse_pages_maps_parse_result(monkeypatch, tmp_path):
    monkeypatch.setattr(adapters, "LiteParse", _FakeLiteParse)

    pages, meta = adapters.extract_liteparse_pages(
        file_path=str(tmp_path / "sample.pdf"),
        screenshot_dir=tmp_path / "shots",
    )

    assert meta["layout_engine"] == "liteparse"
    assert len(pages) == 1
    page = pages[0]
    assert page["page_num"] == 1
    assert page["full_page_text"] == "Revenue increased"
    assert page["screenshot_path"] == "/tmp/page_1.png"
    assert page["text_items"][0]["text"] == "Revenue"
    assert page["ocr_used"] is False


def test_extract_liteparse_pages_uses_page_level_ocr_flag(monkeypatch, tmp_path):
    class _OCRPage(_FakePage):
        def __init__(self):
            super().__init__()
            self.ocrUsed = True

    class _OCRParseResult:
        def __init__(self):
            self.pages = [_OCRPage()]

    class _OCRLiteParse(_FakeLiteParse):
        def parse(self, *_args, **_kwargs):
            return _OCRParseResult()

    monkeypatch.setattr(adapters, "LiteParse", _OCRLiteParse)
    pages, _meta = adapters.extract_liteparse_pages(
        file_path=str(tmp_path / "sample.pdf"),
        screenshot_dir=tmp_path / "shots",
    )

    assert pages[0]["ocr_used"] is True


def test_repair_pdf_path_falls_back_to_original_when_all_methods_fail(
    monkeypatch, tmp_path
):
    file_path = tmp_path / "bad.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(repair, "_try_mutool_clean", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        repair,
        "_try_page_copy_rebuild",
        lambda *_args, **_kwargs: (False, "page-copy-failed"),
    )

    repaired, status = repair.repair_pdf_path(str(file_path))

    assert repaired == str(file_path)
    assert status["pdf_repair_attempted"] is True
    assert status["pdf_repair_method"] == "pymupdf_page_copy"
    assert status["pdf_repair_success"] is False
    assert status["pdf_repair_error"] == "page-copy-failed"


def test_suppress_mupdf_messages_calls_tool_switches(monkeypatch):
    calls = {"errors": None, "warnings": None}

    class _FakeTools:
        def mupdf_display_errors(self, value):
            calls["errors"] = value

        def mupdf_display_warnings(self, value):
            calls["warnings"] = value

    class _FakeFitz:
        TOOLS = _FakeTools()

    monkeypatch.setattr(repair, "fitz", _FakeFitz)
    repair.suppress_mupdf_messages()

    assert calls["errors"] is False
    assert calls["warnings"] is False


def test_analyze_chart_artifacts_populates_structured_chart_fields(monkeypatch):
    class _FakeLLM:
        def complete(self, _prompt):
            return """{
                "chart_type": "line",
                "chart_title": "Revenue Trend",
                "x_axis_label": "Quarter",
                "y_axis_label": "Revenue",
                "x_categories": ["Q1", "Q2", "Q3"],
                "series": ["Revenue"],
                "approx_datapoints": [
                    {"series": "Revenue", "x": "Q1", "y": 10, "unit": "USDm", "approximate": true},
                    {"series": "Revenue", "x": "Q2", "y": 20, "unit": "USDm", "approximate": true},
                    {"series": "Revenue", "x": "Q3", "y": 30, "unit": "USDm", "approximate": true}
                ],
                "trend_summary": "Revenue increases each quarter.",
                "key_chart_facts": ["Q3 is about 3x Q1."],
                "numeric_extraction_confidence": 0.84
            }"""

    manifest = PageManifest(
        document_id="doc-1",
        page_num=1,
        page_width=612,
        page_height=792,
        screenshot_path="",
        full_page_text="",
        layout_confidence=0.8,
        complexity_score=0.9,
        page_class="visual_heavy_page",
        ocr_used=False,
        parser_sources=["liteparse", "pymupdf"],
    )
    figure = FigureArtifact(
        figure_id="fig-1",
        page_num=1,
        bbox=[10, 10, 200, 200],
        figure_type="chart",
        caption_text="Figure 1 Revenue Trend",
        nearby_text="Revenue rises from 10 to 30.",
        section_path="Financials",
        crop_path="/tmp/crop.png",
        page_screenshot_path="",
        visual_proxy_text="",
        footnotes=[],
        confidence=0.8,
    )

    monkeypatch.setattr(settings, "ENABLE_MULTIMODAL_CAPTIONING", True)
    monkeypatch.setattr(settings, "LLM_CAPTION_MAX_PAGES", 3)
    monkeypatch.setattr(settings, "LLM_CAPTION_MAX_ARTIFACTS_PER_PAGE", 3)
    monkeypatch.setattr(artifact_builders, "is_placeholder_mode", lambda: False)
    monkeypatch.setattr(
        artifact_builders,
        "LlamaSettings",
        type("_FakeSettings", (), {"llm": _FakeLLM()}),
    )

    artifact_builders.analyze_chart_artifacts([manifest], [figure])

    assert figure.llm_caption_status == "success"
    assert figure.chart_type == "line"
    assert figure.chart_title == "Revenue Trend"
    assert len(figure.approx_datapoints) == 3
    assert figure.numeric_extraction_confidence == 0.84
    assert figure.llm_enriched is True


def test_analyze_chart_artifacts_falls_back_when_llm_fails(monkeypatch):
    class _FailingLLM:
        def complete(self, _prompt):
            raise RuntimeError("llm error")

    manifest = PageManifest(
        document_id="doc-1",
        page_num=1,
        page_width=612,
        page_height=792,
        screenshot_path="",
        full_page_text="",
        layout_confidence=0.8,
        complexity_score=0.9,
        page_class="visual_heavy_page",
        ocr_used=False,
        parser_sources=["liteparse", "pymupdf"],
    )
    figure = FigureArtifact(
        figure_id="fig-1",
        page_num=1,
        bbox=[10, 10, 200, 200],
        figure_type="chart",
        caption_text="Figure 1 Revenue Trend",
        nearby_text="The series increases from 10 to 30 over time.",
        section_path="Financials",
        crop_path="/tmp/crop.png",
        page_screenshot_path="",
        visual_proxy_text="",
        footnotes=[],
        confidence=0.8,
    )

    monkeypatch.setattr(settings, "ENABLE_MULTIMODAL_CAPTIONING", True)
    monkeypatch.setattr(settings, "LLM_CAPTION_MAX_PAGES", 3)
    monkeypatch.setattr(settings, "LLM_CAPTION_MAX_ARTIFACTS_PER_PAGE", 3)
    monkeypatch.setattr(artifact_builders, "is_placeholder_mode", lambda: False)
    monkeypatch.setattr(
        artifact_builders,
        "LlamaSettings",
        type("_FakeSettings", (), {"llm": _FailingLLM()}),
    )

    artifact_builders.analyze_chart_artifacts([manifest], [figure])

    assert figure.llm_caption_status == "failed"
    assert figure.chart_parse_status in {"partial", "success"}
    assert len(figure.approx_datapoints) >= 2


def test_page_structure_uses_layout_predictions_for_region_typing():
    liteparse_pages = [
        {
            "page_num": 1,
            "page_width": 600,
            "page_height": 800,
            "full_page_text": "Revenue breakdown",
            "text_items": [],
            "ocr_used": False,
            "parser_source": "liteparse",
        }
    ]
    pymupdf_pages = [
        {
            "page_num": 1,
            "width": 600,
            "height": 800,
            "blocks": [{"bbox": [50, 200, 320, 240], "text": "Revenue breakdown"}],
            "table_candidates": [],
            "image_refs": [],
            "vector_count": 0,
            "layout_predictions": [{"bbox": [40, 180, 340, 260], "label": "table"}],
        }
    ]
    parse_meta = {"layout_engine": "pymupdf_layout", "page_parse_degraded": False}
    repair_meta = {}

    manifests, regions = page_structure.build_page_manifests_and_regions(
        document_id="doc-layout-1",
        liteparse_pages=liteparse_pages,
        pymupdf_pages=pymupdf_pages,
        parse_meta=parse_meta,
        repair_meta=repair_meta,
    )

    assert manifests
    assert regions
    assert regions[0].region_type == "table_region"
    assert regions[0].zone == "table_zone"
    assert "pymupdf_layout" in manifests[0].parser_sources
