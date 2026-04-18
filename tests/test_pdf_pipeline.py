from pathlib import Path

from app.core.config import settings
from app.ingestion.pdf_pipeline import (
    adapters,
    artifact_builders,
    page_structure,
    repair,
)
from app.ingestion.pdf_pipeline.models import (
    FigureArtifact,
    PageManifest,
    TableArtifact,
)


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


def test_extract_table_candidates_uses_strategy_cascade():
    class _FakeTable:
        def __init__(self):
            self.bbox = [1, 2, 10, 20]

        def extract(self):
            return [["Quarter", "Revenue"], ["Q1", "10"]]

    class _FakeFinder:
        def __init__(self, tables):
            self.tables = tables

    class _FakePage:
        number = 4

        def __init__(self):
            self.calls = []

        def find_tables(self, **kwargs):
            self.calls.append(kwargs)
            strategy = kwargs.get("strategy", "default")
            if strategy == "default":
                raise RuntimeError("default strategy failed")
            if strategy == "lines":
                return _FakeFinder([])
            return _FakeFinder([_FakeTable()])

    page = _FakePage()
    candidates = adapters._extract_table_candidates(page)

    assert candidates is not None
    assert len(candidates) == 1
    assert candidates[0]["candidate_id"].startswith("text_table_candidate_")
    assert candidates[0]["rows"][0] == ["Quarter", "Revenue"]
    assert page.calls == [{}, {"strategy": "lines"}, {"strategy": "text"}]


def test_extract_table_candidates_returns_none_when_all_strategies_fail():
    class _FakePage:
        number = 1

        def find_tables(self, **_kwargs):
            raise RuntimeError("table detection failed")

    assert adapters._extract_table_candidates(_FakePage()) is None


def test_extract_table_candidates_skips_malformed_bbox_objects():
    class _MalformedBBoxTable:
        @property
        def bbox(self):
            raise ValueError("min() iterable argument is empty")

        def extract(self):
            return [["Quarter", "Revenue"], ["Q1", "10"]]

    class _FakeFinder:
        def __init__(self, tables):
            self.tables = tables

    class _FakePage:
        number = 2

        def __init__(self):
            self.calls = []

        def find_tables(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("strategy", "default") == "default":
                return _FakeFinder([_MalformedBBoxTable()])
            return _FakeFinder([])

    page = _FakePage()
    candidates = adapters._extract_table_candidates(page)

    assert candidates == []
    assert page.calls == [{}, {"strategy": "lines"}, {"strategy": "text"}]


def test_build_table_candidates_keeps_valid_when_first_bbox_is_malformed():
    class _MalformedBBoxTable:
        @property
        def bbox(self):
            raise ValueError("min() iterable argument is empty")

        def extract(self):
            return [["Quarter", "Revenue"], ["Q1", "10"]]

    class _ValidTable:
        bbox = [1, 2, 10, 20]

        def extract(self):
            return [["Quarter", "Revenue"], ["Q2", "20"]]

    candidates = adapters._build_table_candidates_from_tables(
        tables=[_MalformedBBoxTable(), _ValidTable()],
        strategy_name="default",
        page_num=7,
    )

    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == "default_table_candidate_2"
    assert candidates[0]["rows"][1] == ["Q2", "20"]


def test_extract_table_candidates_uses_later_strategy_when_earlier_has_malformed_bbox():
    class _MalformedBBoxTable:
        @property
        def bbox(self):
            raise ValueError("min() iterable argument is empty")

        def extract(self):
            return [["Quarter", "Revenue"], ["Q1", "10"]]

    class _ValidTable:
        bbox = [1, 2, 10, 20]

        def extract(self):
            return [["Quarter", "Revenue"], ["Q3", "30"]]

    class _FakeFinder:
        def __init__(self, tables):
            self.tables = tables

    class _FakePage:
        number = 4

        def __init__(self):
            self.calls = []

        def find_tables(self, **kwargs):
            self.calls.append(kwargs)
            strategy = kwargs.get("strategy", "default")
            if strategy == "default":
                return _FakeFinder([_MalformedBBoxTable()])
            if strategy == "lines":
                return _FakeFinder([])
            return _FakeFinder([_ValidTable()])

    page = _FakePage()
    candidates = adapters._extract_table_candidates(page)

    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == "text_table_candidate_1"
    assert candidates[0]["rows"][1] == ["Q3", "30"]
    assert page.calls == [{}, {"strategy": "lines"}, {"strategy": "text"}]


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
    monkeypatch.setattr(
        artifact_builders,
        "_run_chart_inference",
        lambda _prompt, _image_path: artifact_builders.ChartCaptionResponse(
            chart_type="line",
            chart_title="Revenue Trend",
            x_axis_label="Quarter",
            y_axis_label="Revenue",
            x_categories=["Q1", "Q2", "Q3"],
            series=["Revenue"],
            approx_datapoints=[
                artifact_builders.ChartDatapointResponse(
                    series="Revenue",
                    x="Q1",
                    y=10,
                    unit="USDm",
                    approximate=True,
                ),
                artifact_builders.ChartDatapointResponse(
                    series="Revenue",
                    x="Q2",
                    y=20,
                    unit="USDm",
                    approximate=True,
                ),
                artifact_builders.ChartDatapointResponse(
                    series="Revenue",
                    x="Q3",
                    y=30,
                    unit="USDm",
                    approximate=True,
                ),
            ],
            trend_summary="Revenue increases each quarter.",
            key_chart_facts=["Q3 is about 3x Q1."],
            numeric_extraction_confidence=0.84,
        ),
    )

    artifact_builders.analyze_chart_artifacts([manifest], [figure])

    assert figure.llm_caption_status == "success"
    assert figure.chart_type == "line"
    assert figure.chart_title == "Revenue Trend"
    assert len(figure.approx_datapoints) == 3
    assert figure.numeric_extraction_confidence == 0.84
    assert figure.llm_enriched is True


def test_analyze_chart_artifacts_falls_back_when_llm_fails(monkeypatch):
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
    monkeypatch.setattr(
        artifact_builders,
        "_run_chart_inference",
        lambda _prompt, _image_path: (_ for _ in ()).throw(RuntimeError("llm error")),
    )

    artifact_builders.analyze_chart_artifacts([manifest], [figure])

    assert figure.llm_caption_status == "failed"
    assert figure.chart_parse_status in {"partial", "success"}
    assert len(figure.approx_datapoints) >= 2


def test_run_multimodal_inference_503_returns_none_and_warns(
    monkeypatch, tmp_path, caplog
):
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"fake-image-bytes")
    monkeypatch.setattr(settings, "AI_API_KEY", "test-openai-key")

    class _FakeServerError(Exception):
        def __init__(self, status_code: int, message: str):
            super().__init__(message)
            self.status_code = status_code

    class _FakeLLM:
        def chat(self, _messages):
            raise _FakeServerError(
                503,
                "503 UNAVAILABLE. Deadline expired before operation could complete.",
            )

    monkeypatch.setattr(
        artifact_builders,
        "LlamaSettings",
        type("_FakeSettings", (), {"llm": _FakeLLM()}),
    )

    caplog.set_level("WARNING")
    result = artifact_builders._run_multimodal_inference(
        "Extract chart structure", str(image_path)
    )

    assert result is None
    assert "Multimodal inference unavailable" in caplog.text
    assert "Falling back to text-only inference" in caplog.text


def test_build_reasoning_artifact_handles_non_list_fields(monkeypatch):
    def _fake_inference(_prompt: str, _max_output: int):
        return artifact_builders.ReasoningInferenceResult(
            payload={
                "key_insights": "Revenue rose materially",
                "metric_comparisons": {"q4_vs_q3": "Q4 > Q3"},
                "trend_statement": "Upward momentum continues",
                "caveats": "Values are approximate",
                "evidence_refs": {"table_id": "table-1", "row": "Net Revenue"},
            },
            used_structured_output=False,
        )

    monkeypatch.setattr(artifact_builders, "_run_reasoning_inference", _fake_inference)
    service = artifact_builders.ReasoningEnrichmentService([], [], [])

    artifact = service._build_reasoning_artifact(
        prompt="analyze",
        reasoning_type="page_reasoning",
        page_num=11,
        source_artifact_ids=["table-1"],
        evidence_refs={"source_artifact_ids": ["table-1"]},
        success_confidence=0.8,
        fallback_confidence=0.5,
    )

    assert artifact is not None
    assert artifact.claims == ["Revenue rose materially"]
    assert artifact.evidence_refs["evidence"] == [
        "table_id: table-1",
        "row: Net Revenue",
    ]
    assert "Trend: Upward momentum continues" in artifact.text
    assert "Comparison: q4_vs_q3: Q4 > Q3" in artifact.text
    assert "Caveat: Values are approximate" in artifact.text


def test_run_reasoning_inference_prefers_structured_output(monkeypatch):
    class _FakeLLM:
        def structured_predict(self, *_args, **_kwargs):
            return artifact_builders.ReasoningStructuredResponse(
                key_insights=["Structured insight"],
                metric_comparisons=["Q4 > Q3"],
                trend_statement="Upward trend",
                caveats=["Approximate values"],
                evidence_refs=["table-1: net revenue"],
            )

        def complete(self, _prompt):
            raise AssertionError(
                "complete() should not be called when structured output succeeds"
            )

    monkeypatch.setattr(
        artifact_builders,
        "LlamaSettings",
        type("_FakeSettings", (), {"llm": _FakeLLM()}),
    )

    result = artifact_builders._run_reasoning_inference("analyze", 2000)

    assert result is not None
    assert result.used_structured_output is True
    assert result.payload["key_insights"] == ["Structured insight"]


def test_run_reasoning_inference_uses_user_prompt_kwarg(monkeypatch):
    class _FakeLLM:
        def structured_predict(
            self, output_cls, prompt, llm_kwargs=None, **prompt_args
        ):
            assert output_cls is artifact_builders.ReasoningStructuredResponse
            assert llm_kwargs is None
            assert "prompt" not in prompt_args
            assert prompt_args["user_prompt"] == "analyze this"
            return artifact_builders.ReasoningStructuredResponse(
                key_insights=["Structured insight"]
            )

        def complete(self, _prompt):
            raise AssertionError(
                "complete() should not be called when structured output succeeds"
            )

    monkeypatch.setattr(
        artifact_builders,
        "LlamaSettings",
        type("_FakeSettings", (), {"llm": _FakeLLM()}),
    )

    result = artifact_builders._run_reasoning_inference("analyze this", 2000)

    assert result is not None
    assert result.used_structured_output is True
    assert result.payload["key_insights"] == ["Structured insight"]


def test_run_structured_text_inference_uses_user_prompt_kwarg(monkeypatch):
    class _FakeLLM:
        def structured_predict(
            self, output_cls, prompt, llm_kwargs=None, **prompt_args
        ):
            assert output_cls is artifact_builders.ArtifactEnrichmentResponse
            assert llm_kwargs is None
            assert "prompt" not in prompt_args
            assert prompt_args["user_prompt"] == "summarize table"
            return artifact_builders.ArtifactEnrichmentResponse(
                summary_points=["point"]
            )

    monkeypatch.setattr(
        artifact_builders,
        "LlamaSettings",
        type("_FakeSettings", (), {"llm": _FakeLLM()}),
    )

    result = artifact_builders._run_structured_text_inference(
        "summarize table",
        artifact_builders.ArtifactEnrichmentResponse,
    )

    assert result is not None
    assert result.summary_points == ["point"]


def test_analyze_chart_artifacts_timeout_uses_heuristic_fallback(monkeypatch):
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
        figure_id="fig-timeout-1",
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
    monkeypatch.setattr(artifact_builders, "_llm_available_for_pipeline", lambda: True)

    def _fake_parallel_jobs(*_args, **_kwargs):
        return [
            ((figure, "prompt"), None, TimeoutError("job exceeded timeout (25.0s)"))
        ]

    monkeypatch.setattr(artifact_builders, "_run_parallel_jobs", _fake_parallel_jobs)

    artifact_builders.analyze_chart_artifacts([manifest], [figure])

    assert figure.llm_caption_status == "timeout_fallback"
    assert figure.llm_caption_error == "job exceeded timeout (25.0s)"
    assert figure.chart_parse_status in {"partial", "success"}
    assert len(figure.approx_datapoints) >= 2


def test_maybe_llm_enrich_artifacts_uses_structured_summary_points(monkeypatch):
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
    table = TableArtifact(
        table_id="table-1",
        page_nums=[1],
        bbox_list=[[10.0, 10.0, 200.0, 120.0]],
        caption_text="Revenue table",
        section_path="Financials",
        html_table="<table></table>",
        json_table=[["Quarter", "Revenue"], ["Q1", "10"]],
        normalized_table_text="Quarter Revenue\nQ1 10",
        header_rows=["Quarter", "Revenue"],
        units=["USDm"],
        footnotes=[],
        continuation_flag=False,
        confidence=0.8,
    )
    figure = FigureArtifact(
        figure_id="fig-2",
        page_num=1,
        bbox=[10, 10, 200, 200],
        figure_type="chart",
        caption_text="Revenue trend",
        nearby_text="Revenue rises",
        section_path="Financials",
        crop_path="",
        page_screenshot_path="",
        visual_proxy_text="Initial figure summary",
        footnotes=[],
        confidence=0.8,
    )

    monkeypatch.setattr(settings, "ENABLE_LLM_ARTIFACT_ENRICHMENT", True)
    monkeypatch.setattr(settings, "LLM_ARTIFACT_ENRICHMENT_MAX_PAGES", 3)
    monkeypatch.setattr(artifact_builders, "_llm_available_for_pipeline", lambda: True)

    def _fake_parallel_jobs(jobs, _runner, **_kwargs):
        response = artifact_builders.ArtifactEnrichmentResponse(
            summary_points=[
                "Revenue increases from Q1 to Q4.",
                "All values are reported in USDm.",
            ]
        )
        return [(jobs[0], response, None)]

    monkeypatch.setattr(artifact_builders, "_run_parallel_jobs", _fake_parallel_jobs)

    artifact_builders.maybe_llm_enrich_artifacts([manifest], [table], [figure])

    assert "LLM summary:" in table.normalized_table_text
    assert "- Revenue increases from Q1 to Q4." in table.normalized_table_text
    assert table.llm_enriched is True
    assert "LLM summary:" in figure.visual_proxy_text
    assert "- All values are reported in USDm." in figure.visual_proxy_text
    assert figure.llm_enriched is True


def test_analyze_page_screenshots_uses_structured_output(monkeypatch):
    manifest = PageManifest(
        document_id="doc-1",
        page_num=1,
        page_width=612,
        page_height=792,
        screenshot_path="/tmp/page_1.png",
        full_page_text="",
        layout_confidence=0.8,
        complexity_score=0.9,
        page_class="visual_heavy_page",
        ocr_used=False,
        parser_sources=["liteparse", "pymupdf"],
    )

    monkeypatch.setattr(settings, "ENABLE_MULTIMODAL_CAPTIONING", True)
    monkeypatch.setattr(settings, "LLM_CAPTION_MAX_PAGES", 3)
    monkeypatch.setattr(artifact_builders, "_llm_available_for_pipeline", lambda: True)

    def _fake_parallel_jobs(jobs, _runner, **_kwargs):
        response = artifact_builders.PageScreenshotResponse(
            layout_description="Two-column page with a chart and table.",
            numeric_values=["Revenue: 10, 20, 30 USDm"],
            chart_descriptions=["Line chart trending upward from Q1 to Q3."],
            table_summaries=["Table lists quarterly revenue and margin."],
            map_or_diagram_annotations=["No maps present."],
            key_takeaways=["Revenue growth is consistent across quarters."],
        )
        return [(jobs[0], response, None)]

    monkeypatch.setattr(artifact_builders, "_run_parallel_jobs", _fake_parallel_jobs)

    artifact_builders.analyze_page_screenshots([manifest])

    assert manifest.llm_enriched is True
    assert manifest.llm_page_summary is not None
    assert manifest.llm_page_summary_status == "success"
    assert manifest.llm_page_summary_error is None
    assert (
        "Layout: Two-column page with a chart and table." in manifest.llm_page_summary
    )
    assert "Numeric values:" in manifest.llm_page_summary
    assert "Charts:" in manifest.llm_page_summary


def test_analyze_page_screenshots_retries_then_succeeds(monkeypatch):
    manifest = PageManifest(
        document_id="doc-1",
        page_num=5,
        page_width=612,
        page_height=792,
        screenshot_path="/tmp/page_5.png",
        full_page_text="Revenue rose from 10 to 20.",
        layout_confidence=0.7,
        complexity_score=0.85,
        page_class="visual_heavy_page",
        ocr_used=False,
        parser_sources=["liteparse", "pymupdf"],
    )

    monkeypatch.setattr(settings, "ENABLE_MULTIMODAL_CAPTIONING", True)
    monkeypatch.setattr(settings, "LLM_CAPTION_MAX_PAGES", 3)
    monkeypatch.setattr(settings, "LLM_SCREENSHOT_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(settings, "LLM_SCREENSHOT_MAX_WORKERS", 1)
    monkeypatch.setattr(settings, "LLM_SCREENSHOT_RETRIES", 1)
    monkeypatch.setattr(artifact_builders, "_llm_available_for_pipeline", lambda: True)

    calls = {"count": 0}

    def _fake_parallel_jobs(jobs, _runner, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return [(jobs[0], None, TimeoutError("job exceeded timeout (2.0s)"))]
        response = artifact_builders.PageScreenshotResponse(
            layout_description="Second attempt succeeds.",
            numeric_values=["Revenue: 20"],
            chart_descriptions=[],
            table_summaries=[],
            map_or_diagram_annotations=[],
            key_takeaways=["Recovered after retry."],
        )
        return [(jobs[0], response, None)]

    monkeypatch.setattr(artifact_builders, "_run_parallel_jobs", _fake_parallel_jobs)

    artifact_builders.analyze_page_screenshots([manifest])

    assert calls["count"] == 2
    assert manifest.llm_page_summary_status == "success"
    assert manifest.llm_page_summary_error is None
    assert manifest.llm_enriched is True
    assert "Second attempt succeeds." in (manifest.llm_page_summary or "")


def test_analyze_page_screenshots_falls_back_after_retry_budget(monkeypatch):
    manifest = PageManifest(
        document_id="doc-1",
        page_num=8,
        page_width=612,
        page_height=792,
        screenshot_path="/tmp/page_8.png",
        full_page_text="Revenue was 10, 20, and 30 percent in key regions.",
        layout_confidence=0.65,
        complexity_score=0.9,
        page_class="hard_page",
        ocr_used=False,
        parser_sources=["liteparse", "pymupdf"],
    )

    monkeypatch.setattr(settings, "ENABLE_MULTIMODAL_CAPTIONING", True)
    monkeypatch.setattr(settings, "LLM_CAPTION_MAX_PAGES", 3)
    monkeypatch.setattr(settings, "LLM_SCREENSHOT_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(settings, "LLM_SCREENSHOT_MAX_WORKERS", 1)
    monkeypatch.setattr(settings, "LLM_SCREENSHOT_RETRIES", 1)
    monkeypatch.setattr(artifact_builders, "_llm_available_for_pipeline", lambda: True)

    def _fake_parallel_jobs(jobs, _runner, **_kwargs):
        return [(jobs[0], None, TimeoutError("job exceeded timeout (1.0s)"))]

    monkeypatch.setattr(artifact_builders, "_run_parallel_jobs", _fake_parallel_jobs)

    artifact_builders.analyze_page_screenshots([manifest])

    assert manifest.llm_page_summary_status == "fallback"
    assert "timeout" in (manifest.llm_page_summary_error or "").lower()
    assert manifest.llm_enriched is True
    assert "Deterministic parser fallback summary" in (manifest.llm_page_summary or "")
    assert "Numeric values:" in (manifest.llm_page_summary or "")


def test_parse_chart_json_response_rejects_non_object_json():
    assert artifact_builders._parse_chart_json_response('["a", "b"]') is None


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
