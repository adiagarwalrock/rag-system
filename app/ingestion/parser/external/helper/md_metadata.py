"""
MarkdownPageAnalyzer — base metadata layer from markdown heuristics.

Produces a populated ChunkMetadata for each page section coming out of
parse(). Fields that require document-level LLM reasoning (document_date,
as_of_date, metric_basis, chart_* enrichment, etc.) are left at their
Pydantic defaults; the provider Extract API fills those in.

The analyzer also classifies chunk_type from content signals — this is the
only place chunk_type is assigned. The Extract API never modifies it.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.parser.custom.pdf_pipeline.helpers import (
    extract_units,
    has_chart_signals,
    has_table_signals,
    numeric_density,
)

# ---------------------------------------------------------------------------
# Heading regex
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_PIPE_TABLE_ROW_RE = re.compile(r"^\|.+\|")
_SEPARATOR_ROW_RE = re.compile(r"^\|[-| :]+\|$")

# Chunk types mirror the custom pipeline's artifact type taxonomy so that
# downstream metadata exclusion lists, retrieval filters, and evaluation
# queries can use the same values across both paths.
CHUNK_TYPES = {
    "full_table",
    "table_segment",
    "table_summary_text",
    "figure_artifact",
    "visual_proxy_text",
    "chart_context",
    "chart_data_points",
    "body_text",
    "page_card",
    "reasoning",
}


@dataclass
class AnalyzedChunk:
    """Result of analyzing one page section of markdown."""

    chunk_id: str
    chunk_type: str
    source_artifact_type: str
    source_artifact_id: str
    page_nums: list[int]
    text: str
    metadata: dict[str, Any]
    asset_refs: list[str] = field(default_factory=list)


class MarkdownPageAnalyzer:
    """Classify and extract base metadata from one page section of markdown.

    Instantiate once per parser class, reuse across many pages.

    Parameters
    ----------
    parser_name:
        Short name of the external parser, e.g. ``"reducto"`` or
        ``"llamaparse"``. Stored in ``parser_sources``.
    layout_engine:
        Identifier for the VLM/layout engine, e.g. ``"reducto_vlm"`` or
        ``"llamaparse_vlm"``. Stored in ``layout_engine``.
    """

    def __init__(self, *, parser_name: str, layout_engine: str) -> None:
        self.parser_name = parser_name
        self.layout_engine = layout_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, page_num: int, content: str) -> AnalyzedChunk:
        """Analyse one page section and return an :class:`AnalyzedChunk`.

        The chunk_type is determined from markdown content signals.
        All base metadata fields are populated; temporal / chart enrichment
        fields are left at None for the Extract API to fill.
        """
        chunk_type, source_artifact_type = self._classify(content)
        source_artifact_id = f"{source_artifact_type}_{page_num}_{uuid.uuid4().hex[:8]}"
        chunk_id = f"p{page_num}_{source_artifact_id}"

        metadata = self._build_metadata(
            page_num=page_num,
            content=content,
            chunk_type=chunk_type,
            source_artifact_type=source_artifact_type,
            source_artifact_id=source_artifact_id,
            chunk_id=chunk_id,
        )

        return AnalyzedChunk(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            source_artifact_type=source_artifact_type,
            source_artifact_id=source_artifact_id,
            page_nums=[page_num],
            text=content,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, content: str) -> tuple[str, str]:
        """Return (chunk_type, source_artifact_type) based on content signals."""
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if not lines:
            return "body_text", "body_text"

        pipe_rows = sum(
            1
            for ln in lines
            if _PIPE_TABLE_ROW_RE.match(ln) and not _SEPARATOR_ROW_RE.match(ln)
        )
        total_lines = len(lines)
        table_ratio = pipe_rows / total_lines if total_lines else 0

        has_chart = has_chart_signals(content)
        has_table = has_table_signals(content)

        # Figure / chart: chart signals dominate and table content is minimal
        if has_chart and table_ratio < 0.3:
            return "figure_artifact", "figure_artifact"

        # Pure or near-pure table
        if table_ratio >= 0.5 or (has_table and table_ratio >= 0.3):
            return "full_table", "full_table"

        # Mixed — table present but also prose
        if has_table and table_ratio > 0:
            return "table_summary_text", "full_table"

        return "body_text", "body_text"

    # ------------------------------------------------------------------
    # Metadata builder
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        *,
        page_num: int,
        content: str,
        chunk_type: str,
        source_artifact_type: str,
        source_artifact_id: str,
        chunk_id: str,
    ) -> dict[str, Any]:
        section_path = self._extract_section_path(content)
        caption = self._extract_caption(content)
        nd = numeric_density(content)
        complexity = self._complexity_score(chunk_type, nd)

        if chunk_type in {"full_table", "table_segment"}:
            page_class = "table_heavy_page"
        elif chunk_type in {"figure_artifact", "visual_proxy_text", "chart_context", "chart_data_points"}:
            page_class = "visual_heavy_page"
        else:
            page_class = "simple_text_page"

        units_list = extract_units(content)

        approx_datapoints = self._extract_approx_datapoints(content)

        return {
            # ── identity ─────────────────────────────────────────────
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "source_artifact_type": source_artifact_type,
            "source_artifact_id": source_artifact_id,
            "page_num": page_num,
            "page_nums": [page_num],
            # ── layout / quality ──────────────────────────────────────
            "section_path": section_path,
            "caption": caption,
            "numeric_density": nd,
            "layout_confidence": 1.0,
            "complexity_score": complexity,
            "parser_sources": [self.parser_name],
            "page_class": page_class,
            "layout_engine": self.layout_engine,
            "ocr_used": False,
            "units": units_list,
            "continuation_flag": False,
            "page_parse_degraded": False,
            "degraded_stages": [],
            "pdf_repair_attempted": False,
            "pdf_repair_method": None,
            "pdf_repair_success": False,
            "pdf_repair_error": None,
            # ── LLM enrichment status ─────────────────────────────────
            "llm_page_summary_status": "not_run",
            "llm_page_summary_error": None,
            "llm_caption_status": "skipped",
            "llm_caption_error": None,
            "llm_caption_model": None,
            "llm_caption_version": None,
            "llm_caption_prompt_version": None,
            "chart_parse_status": "not_applicable",
            "llm_enriched": True,          # enriched via Extract API
            "llm_enrichment_confidence": None,
            # ── chart metadata (filled by Extract API) ────────────────
            "chart_type": None,
            "chart_title": None,
            "x_axis_label": None,
            "y_axis_label": None,
            "x_categories": [],
            "series": [],
            "approx_datapoints": approx_datapoints,
            "trend_summary": None,
            "key_chart_facts": [],
            "numeric_extraction_confidence": None,
            # ── table metadata (filled by Extract API) ────────────────
            "table_id": None,
            "table_title": None,
            # ── temporal / scope (filled by Extract API) ─────────────
            "document_date": None,
            "as_of_date": None,
            "metric_basis": None,
            # ── parser tracking ───────────────────────────────────────
            "parser_name": self.parser_name,
            "parser_version": None,   # set by parser class after construction
            # ── misc ──────────────────────────────────────────────────
            "region_ids": [],
            "bbox_refs": [],
            "asset_refs": [],
            "figure_type": None,
            "source_artifact_ids": [],
            "evidence_refs": [],
            "reasoning_confidence": None,
            "reasoning_model": None,
            "reasoning_prompt_version": None,
            "reasoning_type": None,
            "claims": [],
            "artifact_bundle_path": "",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_section_path(self, content: str) -> str:
        """Return headings joined as 'H1 > H2 > H3', matching the custom pipeline's str format."""
        headings: list[str] = []
        for line in content.splitlines():
            m = _HEADING_RE.match(line.strip())
            if m:
                headings.append(m.group(2).strip())
        return " > ".join(headings)

    def _extract_caption(self, content: str) -> str | None:
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _HEADING_RE.match(stripped):
                continue
            if len(stripped) <= 200:
                return stripped
            return stripped[:200]
        return None

    def _complexity_score(self, chunk_type: str, nd: float) -> float:
        if chunk_type in {"figure_artifact", "chart_context", "chart_data_points"}:
            return 0.9
        if chunk_type in {"full_table", "table_segment", "table_summary_text"}:
            return 0.7
        return 0.3

    def _extract_approx_datapoints(
        self, content: str
    ) -> list[dict[str, Any]]:
        """Best-effort extraction of numeric data points from pipe tables."""
        rows: list[list[str]] = []
        header: list[str] | None = None

        for line in content.splitlines():
            stripped = line.strip()
            if not _PIPE_TABLE_ROW_RE.match(stripped):
                continue
            if _SEPARATOR_ROW_RE.match(stripped):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if header is None:
                header = cells
            else:
                rows.append(cells)

        if not header or not rows:
            return []

        datapoints: list[dict[str, Any]] = []
        _NUM = re.compile(r"[-+]?\d[\d,.]*%?")

        for row in rows:
            if not row:
                continue
            label = row[0] if row else ""
            for col_idx, cell in enumerate(row[1:], start=1):
                nums = _NUM.findall(cell)
                if not nums:
                    continue
                col_header = header[col_idx] if col_idx < len(header) else f"col_{col_idx}"
                for num_str in nums:
                    is_pct = num_str.endswith("%")
                    clean = num_str.rstrip("%").replace(",", "")
                    try:
                        value = float(clean)
                    except ValueError:
                        continue
                    datapoints.append(
                        {
                            "series": label,
                            "x": col_header,
                            "y": value,
                            "unit": "%" if is_pct else None,
                            "approximate": False,
                        }
                    )
        return datapoints
