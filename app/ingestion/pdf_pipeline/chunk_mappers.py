from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from typing import Any

from app.core.config import settings
from app.ingestion.pdf_pipeline.helpers import numeric_density, table_to_markdown
from app.ingestion.pdf_pipeline.models import (
    ChunkArtifact,
    FigureArtifact,
    PageManifest,
    ReasoningArtifact,
    TableArtifact,
)
from app.ingestion.pdf_pipeline.registry import PDFPipelineRegistry

logger = logging.getLogger(__name__)


class ArtifactChunkMapper:
    def __init__(self, manifests: dict[int, PageManifest]) -> None:
        self._manifests = manifests

    def table_to_chunks(self, table: TableArtifact) -> list[ChunkArtifact]:
        rows = table.json_table or []
        markdown = table_to_markdown(rows)
        normalized_text = table.normalized_table_text or markdown or table.caption_text
        if not normalized_text.strip():
            return []

        primary_page = table.page_nums[0] if table.page_nums else 1
        manifest = self._manifests.get(primary_page)
        metadata = _base_chunk_metadata(
            manifest=manifest,
            section_path=table.section_path,
            region_ids=[],
            bbox_refs=table.bbox_list,
            caption=table.caption_text,
            units=table.units,
            continuation_flag=table.continuation_flag,
            figure_type=None,
            source_artifact_type="table",
        )
        metadata["table_id"] = table.table_id
        metadata["table_title"] = table.caption_text
        metadata["llm_enriched"] = table.llm_enriched
        metadata["llm_enrichment_confidence"] = table.llm_enrichment_confidence
        metadata["image_scope"] = "page_screenshot"

        table_asset_refs: list[str] = []
        if manifest and manifest.screenshot_path:
            table_asset_refs.append(manifest.screenshot_path)

        table_chunks = [normalized_text]
        if markdown and len(rows) > settings.TABLE_CHUNK_ROW_THRESHOLD:
            try:
                table_chunker = PDFPipelineRegistry().chunker_manager.create_table_chunker(
                    chunk_size=settings.TABLE_CHUNK_ROW_THRESHOLD,
                )
                if table_chunker is not None:
                    pieces = _normalize_chunk_texts(table_chunker.chunk(markdown))
                    if pieces:
                        table_chunks = pieces
            except Exception:
                logger.exception("TableChunker failed for table %s", table.table_id)

        chunks: list[ChunkArtifact] = []
        for idx, text in enumerate(table_chunks, start=1):
            chunk_type = "table_segment" if len(table_chunks) > 1 else "full_table"
            chunks.append(
                ChunkArtifact(
                    chunk_id=str(uuid.uuid4()),
                    chunk_type=chunk_type,
                    source_artifact_type="table",
                    source_artifact_id=f"{table.table_id}:{idx}",
                    page_nums=table.page_nums,
                    text=text,
                    metadata=metadata,
                    asset_refs=table_asset_refs,
                )
            )

        summary_lines = [
            (
                f"Table summary: {table.caption_text}"
                if table.caption_text
                else "Table summary"
            ),
            f"Pages: {', '.join(str(num) for num in table.page_nums)}",
            f"Columns: {', '.join(table.header_rows[:8])}" if table.header_rows else "",
            f"Units: {', '.join(table.units)}" if table.units else "",
        ]
        chunks.append(
            ChunkArtifact(
                chunk_id=str(uuid.uuid4()),
                chunk_type="table_summary_text",
                source_artifact_type="table",
                source_artifact_id=table.table_id,
                page_nums=table.page_nums,
                text="\n".join(line for line in summary_lines if line),
                metadata=metadata,
            )
        )
        return chunks

    def reasoning_to_chunks(
        self,
        reasoning: ReasoningArtifact,
        figures_by_id: dict[str, FigureArtifact] | None = None,
    ) -> list[ChunkArtifact]:
        if not reasoning.text.strip():
            return []

        primary_page = reasoning.page_nums[0] if reasoning.page_nums else 1
        manifest = self._manifests.get(primary_page)

        chunk_type_map = {
            "table_reasoning": "reasoning_table",
            "chart_reasoning": "reasoning_chart",
            "figure_reasoning": "reasoning_figure",
            "page_reasoning": "reasoning_page",
        }
        chunk_type = chunk_type_map.get(reasoning.reasoning_type, "reasoning_page")

        metadata = _base_chunk_metadata(
            manifest=manifest,
            section_path="Reasoning Enrichment",
            region_ids=[],
            bbox_refs=[],
            caption="",
            units=[],
            continuation_flag=False,
            figure_type=None,
            source_artifact_type="reasoning",
        )
        metadata["reasoning_type"] = reasoning.reasoning_type
        metadata["source_artifact_ids"] = reasoning.source_artifact_ids
        metadata["evidence_refs"] = reasoning.evidence_refs
        metadata["reasoning_confidence"] = reasoning.confidence
        metadata["reasoning_model"] = reasoning.model
        metadata["reasoning_prompt_version"] = reasoning.prompt_version
        metadata["llm_enriched"] = True
        metadata["claims"] = reasoning.claims

        asset_refs: list[str] = []
        if figures_by_id:
            for src_id in reasoning.source_artifact_ids or []:
                fig = figures_by_id.get(src_id)
                if fig:
                    ref = fig.crop_path or fig.page_screenshot_path
                    if ref and ref not in asset_refs:
                        asset_refs.append(ref)
        if reasoning.reasoning_type == "page_reasoning":
            if (
                manifest
                and manifest.screenshot_path
                and manifest.screenshot_path not in asset_refs
            ):
                asset_refs.append(manifest.screenshot_path)

        if asset_refs:
            metadata["image_scope"] = (
                "figure_crop"
                if any("crop" in r or "figure" in r for r in asset_refs)
                else "page_screenshot"
            )

        return [
            ChunkArtifact(
                chunk_id=str(uuid.uuid4()),
                chunk_type=chunk_type,
                source_artifact_type="reasoning",
                source_artifact_id=reasoning.reasoning_id,
                page_nums=reasoning.page_nums,
                text=reasoning.text,
                metadata=metadata,
                asset_refs=_dedupe_asset_refs(asset_refs),
            )
        ]

    def figure_to_chunks(self, figure: FigureArtifact) -> list[ChunkArtifact]:
        manifest = self._manifests.get(figure.page_num)
        metadata = _base_chunk_metadata(
            manifest=manifest,
            section_path=figure.section_path,
            region_ids=[],
            bbox_refs=[figure.bbox],
            caption=figure.caption_text,
            units=[],
            continuation_flag=False,
            figure_type=figure.figure_type,
            source_artifact_type="figure",
        )
        metadata.update(
            {
                "llm_enriched": figure.llm_enriched,
                "llm_enrichment_confidence": figure.llm_enrichment_confidence,
                "chart_type": figure.chart_type,
                "chart_title": figure.chart_title,
                "x_axis_label": figure.x_axis_label,
                "y_axis_label": figure.y_axis_label,
                "x_categories": figure.x_categories,
                "series": figure.series,
                "legend_items": figure.legend_items,
                "approx_datapoints": figure.approx_datapoints,
                "trend_summary": figure.trend_summary,
                "key_chart_facts": figure.key_chart_facts,
                "numeric_extraction_confidence": figure.numeric_extraction_confidence,
                "chart_parse_status": figure.chart_parse_status,
                "llm_caption_model": figure.llm_caption_model,
                "llm_caption_version": figure.llm_caption_version,
                "llm_caption_prompt_version": figure.llm_caption_prompt_version,
                "llm_caption_status": figure.llm_caption_status,
                "llm_caption_error": figure.llm_caption_error,
            }
        )

        preferred_assets = (
            [figure.crop_path] if figure.crop_path else [figure.page_screenshot_path]
        )
        metadata["image_scope"] = "figure_crop" if figure.crop_path else "page_screenshot"
        assets = _dedupe_asset_refs(preferred_assets)

        chunks = [
            ChunkArtifact(
                chunk_id=str(uuid.uuid4()),
                chunk_type="figure_artifact",
                source_artifact_type="figure",
                source_artifact_id=figure.figure_id,
                page_nums=[figure.page_num],
                text=f"{figure.caption_text}\n{figure.nearby_text}".strip()
                or figure.visual_proxy_text,
                metadata=metadata,
                asset_refs=assets,
            ),
            ChunkArtifact(
                chunk_id=str(uuid.uuid4()),
                chunk_type="visual_proxy_text",
                source_artifact_type="figure",
                source_artifact_id=figure.figure_id,
                page_nums=[figure.page_num],
                text=figure.visual_proxy_text,
                metadata=metadata,
                asset_refs=assets,
            ),
        ]

        is_chart_like = figure.figure_type in {"chart", "diagram", "infographic"}

        if is_chart_like and figure.nearby_text:
            chart_context_text = "\n".join(
                part for part in [figure.trend_summary, figure.nearby_text] if part
            ).strip()
            chunks.append(
                ChunkArtifact(
                    chunk_id=str(uuid.uuid4()),
                    chunk_type="chart_context",
                    source_artifact_type="figure",
                    source_artifact_id=figure.figure_id,
                    page_nums=[figure.page_num],
                    text=chart_context_text or figure.nearby_text,
                    metadata=metadata,
                    asset_refs=assets,
                )
            )

        has_chart_payload = bool(
            figure.approx_datapoints
            or figure.key_chart_facts
            or figure.chart_title
            or figure.trend_summary
            or figure.chart_parse_status in {"partial", "success"}
        )
        if is_chart_like and has_chart_payload:
            chunks.append(
                ChunkArtifact(
                    chunk_id=str(uuid.uuid4()),
                    chunk_type="chart_data_points",
                    source_artifact_type="figure",
                    source_artifact_id=figure.figure_id,
                    page_nums=[figure.page_num],
                    text=_format_chart_datapoints_text(figure),
                    metadata=metadata,
                    asset_refs=assets,
                )
            )

        if is_chart_like and figure.legend_items:
            legend_text = (
                f"Chart legend — {figure.chart_title or figure.caption_text}:\n"
                + "\n".join(figure.legend_items)
            )
            chunks.append(
                ChunkArtifact(
                    chunk_id=str(uuid.uuid4()),
                    chunk_type="chart_legend_chunk",
                    source_artifact_type="figure",
                    source_artifact_id=figure.figure_id,
                    page_nums=[figure.page_num],
                    text=legend_text,
                    metadata=metadata,
                    asset_refs=assets,
                )
            )

        return chunks


def _base_chunk_metadata(
    manifest: PageManifest | None,
    section_path: str,
    region_ids: list[str],
    bbox_refs: list[list[float]],
    caption: str,
    units: list[str],
    continuation_flag: bool,
    figure_type: str | None,
    source_artifact_type: str,
) -> dict[str, Any]:
    return {
        "section_path": section_path,
        "region_ids": region_ids,
        "bbox_refs": bbox_refs,
        "caption": caption,
        "numeric_density": numeric_density(caption),
        "layout_confidence": manifest.layout_confidence if manifest else 0.0,
        "complexity_score": manifest.complexity_score if manifest else 0.0,
        "asset_refs": [],
        "parser_sources": manifest.parser_sources if manifest else ["unknown"],
        "page_class": manifest.page_class if manifest else "unknown",
        "layout_engine": manifest.layout_engine if manifest else "pymupdf_native",
        "page_parse_degraded": manifest.page_parse_degraded if manifest else False,
        "degraded_stages": manifest.degraded_stages if manifest else [],
        "pdf_repair_attempted": manifest.pdf_repair_attempted if manifest else False,
        "pdf_repair_method": manifest.pdf_repair_method if manifest else None,
        "pdf_repair_success": manifest.pdf_repair_success if manifest else False,
        "pdf_repair_error": manifest.pdf_repair_error if manifest else None,
        "llm_page_summary_status": (
            manifest.llm_page_summary_status if manifest else "not_run"
        ),
        "llm_page_summary_error": (
            manifest.llm_page_summary_error if manifest else None
        ),
        "ocr_used": manifest.ocr_used if manifest else False,
        "units": units,
        "continuation_flag": continuation_flag,
        "figure_type": figure_type,
        "source_artifact_type": source_artifact_type,
        "slide_purpose": _classify_slide_purpose(manifest),
        "visible_page_num": _extract_visible_page_num(manifest),
    }


_OVERVIEW_HEADING_KEYWORDS = (
    "quick facts",
    "fast facts",
    "at a glance",
    "key stats",
    "key facts",
    "headline stats",
    "overview",
    "portfolio highlights",
    "company highlights",
    "investment highlights",
    "summary statistics",
)


def _classify_slide_purpose(manifest: PageManifest | None) -> str:
    if manifest is None:
        return "narrative"
    page_text_lower = (manifest.full_page_text or "").lower()
    if any(kw in page_text_lower for kw in _OVERVIEW_HEADING_KEYWORDS):
        return "overview_stats"
    if manifest.page_class == "visual_heavy_page":
        return "chart_analysis"
    if manifest.page_class == "table_heavy_page":
        return "table_data"
    page_text = manifest.full_page_text or ""
    if len(page_text) < 800 and numeric_density(page_text) > 0.08:
        return "overview_stats"
    return "narrative"


def _extract_visible_page_num(manifest: PageManifest | None) -> int | None:
    if manifest is None:
        return None
    return manifest.page_num if manifest.page_num > 0 else None


def _normalize_chunk_texts(parts: Iterable[Any]) -> list[str]:
    values: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text = part.strip()
        else:
            text = str(getattr(part, "text", "")).strip()
        if text:
            values.append(text)
    return values


def _dedupe_asset_refs(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _format_chart_datapoints_text(figure: FigureArtifact) -> str:
    lines = [
        f"Chart type: {figure.chart_type or figure.figure_type}",
        f"Title: {figure.chart_title or figure.caption_text}",
        f"X axis: {figure.x_axis_label}",
        f"Y axis: {figure.y_axis_label}",
        ("Series: " + ", ".join(figure.series) if figure.series else "Series: unknown"),
    ]
    if figure.trend_summary:
        lines.append(f"Trend: {figure.trend_summary}")
    if figure.key_chart_facts:
        lines.append("Key facts:")
        for fact in figure.key_chart_facts[:8]:
            lines.append(f"- {fact}")
    if figure.approx_datapoints:
        lines.append("Approximate datapoints:")
        for point in figure.approx_datapoints[:40]:
            series = point.get("series", "unknown_series")
            x_val = point.get("x", "")
            y_val = point.get("y")
            unit = point.get("unit", "")
            approx = point.get("approximate", True)
            suffix = " (approx)" if approx else ""
            unit_text = f" {unit}" if unit else ""
            lines.append(f"- {series} | x={x_val} | y={y_val}{unit_text}{suffix}")
    else:
        lines.append(
            "Datapoints: not extractable from text — "
            "refer to the attached image to read bar heights or line positions. "
            "Estimate values visually from the axis scale if needed."
        )
    if figure.numeric_extraction_confidence is not None:
        lines.append(
            f"Numeric extraction confidence: {figure.numeric_extraction_confidence:.2f}"
        )
    lines.append("Note: numeric values may be approximate.")
    return "\n".join(line for line in lines if line.strip())
