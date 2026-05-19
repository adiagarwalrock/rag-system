from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llama_index.core import Document as LlamaDocument

from app.core.config import settings
from app.core.token_budget import count_tokens, get_token_encoding
from app.ingestion.pdf_pipeline.contracts import ChunkStage
from app.ingestion.pdf_pipeline.helpers import (
    dump_json,
    has_numeric_data,
    numeric_density,
    table_to_markdown,
)
from app.ingestion.pdf_pipeline.models import (
    ChunkArtifact,
    FigureArtifact,
    PageManifest,
    ReasoningArtifact,
    Region,
    TableArtifact,
)
from app.ingestion.pdf_pipeline.registry import PDFPipelineRegistry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkBuildInputs:
    page_manifests: list[PageManifest]
    regions: list[Region]
    tables: list[TableArtifact]
    figures: list[FigureArtifact]
    reasoning_artifacts: list[ReasoningArtifact]


class ChunkArtifactAssembler:
    def __init__(self, inputs: ChunkBuildInputs) -> None:
        self._inputs = inputs
        self._manifests = {
            manifest.page_num: manifest for manifest in inputs.page_manifests
        }
        self._regions_by_page = _regions_by_page(inputs.regions)

    def build(self) -> list[ChunkArtifact]:
        chunks: list[ChunkArtifact] = []
        self._append_body_text_chunks(chunks)
        self._append_table_chunks(chunks)
        self._append_figure_chunks(chunks)
        self._append_page_cards(chunks)
        self._append_reasoning_chunks(chunks)
        return chunks

    def _append_body_text_chunks(self, chunks: list[ChunkArtifact]) -> None:
        all_page_nums = set(self._regions_by_page.keys()) | set(self._manifests.keys())
        for page_num in sorted(all_page_nums):
            manifest = self._manifests.get(page_num)
            if manifest is None:
                continue

            page_regions = self._regions_by_page.get(page_num, [])
            body_regions = [
                region
                for region in page_regions
                if region.region_type
                in {"body_text", "section_heading", "subsection_heading"}
            ]
            grouped_regions = _group_regions_by_section(body_regions)
            page_chunk_count = 0
            for section_path, section_regions in grouped_regions.items():
                merged_text = "\n".join(
                    region.text for region in section_regions if region.text
                ).strip()
                if not merged_text:
                    continue
                for split_idx, part in enumerate(split_body_text(merged_text), start=1):
                    chunks.append(
                        ChunkArtifact(
                            chunk_id=str(uuid.uuid4()),
                            chunk_type="body_text",
                            source_artifact_type="region_group",
                            source_artifact_id=f"{page_num}:{section_path}:{split_idx}",
                            page_nums=[page_num],
                            text=part,
                            metadata=_base_chunk_metadata(
                                manifest=manifest,
                                section_path=section_path,
                                region_ids=[
                                    region.region_id for region in section_regions
                                ],
                                bbox_refs=[region.bbox for region in section_regions],
                                caption="",
                                units=[],
                                continuation_flag=False,
                                figure_type=None,
                                source_artifact_type="region_group",
                            ),
                        )
                    )
                    page_chunk_count += 1

            if page_chunk_count == 0 and settings.ENABLE_MULTIMODAL_CAPTIONING:
                llm_summary = getattr(manifest, "llm_page_summary", None)
                if llm_summary and llm_summary.strip():
                    chunks.append(
                        ChunkArtifact(
                            chunk_id=str(uuid.uuid4()),
                            chunk_type="body_text",
                            source_artifact_type="vision_summary",
                            source_artifact_id=f"{page_num}:vision_summary",
                            page_nums=[page_num],
                            text=llm_summary.strip(),
                            metadata=_base_chunk_metadata(
                                manifest=manifest,
                                section_path="vision_extracted",
                                region_ids=[],
                                bbox_refs=[],
                                caption="",
                                units=[],
                                continuation_flag=False,
                                figure_type=None,
                                source_artifact_type="vision_summary",
                            ),
                        )
                    )

    def _append_table_chunks(self, chunks: list[ChunkArtifact]) -> None:
        for table in self._inputs.tables:
            chunks.extend(_table_to_chunks(table, self._manifests))

    def _append_figure_chunks(self, chunks: list[ChunkArtifact]) -> None:
        for figure in self._inputs.figures:
            chunks.extend(_figure_to_chunks(figure, self._manifests))

    def _append_page_cards(self, chunks: list[ChunkArtifact]) -> None:
        tables_by_page: dict[int, list[TableArtifact]] = defaultdict(list)
        for table in self._inputs.tables:
            for page_num in table.page_nums:
                tables_by_page[page_num].append(table)

        figures_by_page: dict[int, list[FigureArtifact]] = defaultdict(list)
        for figure in self._inputs.figures:
            figures_by_page[figure.page_num].append(figure)

        for manifest in self._inputs.page_manifests:
            page_tables = tables_by_page.get(manifest.page_num, [])
            page_figures = figures_by_page.get(manifest.page_num, [])
            if (
                manifest.page_class == "simple_text_page"
                and not page_tables
                and not page_figures
            ):
                continue

            summary = [
                f"Page {manifest.page_num} summary",
                f"Class: {manifest.page_class}",
                f"Layout confidence: {manifest.layout_confidence:.2f}",
                f"Tables: {len(page_tables)}",
                f"Figures: {len(page_figures)}",
            ]
            if page_tables:
                summary.append(
                    "Table captions: "
                    + "; ".join(
                        table.caption_text or table.table_id
                        for table in page_tables[:3]
                    )
                )
            if page_figures:
                summary.append(
                    "Figure captions: "
                    + "; ".join(
                        figure.caption_text or figure.figure_type
                        for figure in page_figures[:3]
                    )
                )

            llm_summary = getattr(manifest, "llm_page_summary", None)
            if llm_summary:
                summary.append("\n--- Holistic Page Analysis ---")
                summary.append(llm_summary)

            chunks.append(
                ChunkArtifact(
                    chunk_id=str(uuid.uuid4()),
                    chunk_type="page_card",
                    source_artifact_type="page",
                    source_artifact_id=f"page_{manifest.page_num}",
                    page_nums=[manifest.page_num],
                    text="\n".join(summary),
                    metadata=_base_chunk_metadata(
                        manifest=manifest,
                        section_path="Page Summary",
                        region_ids=[],
                        bbox_refs=[],
                        caption="",
                        units=[],
                        continuation_flag=False,
                        figure_type=None,
                        source_artifact_type="page",
                    ),
                    asset_refs=(
                        [manifest.screenshot_path] if manifest.screenshot_path else []
                    ),
                )
            )

    def _append_reasoning_chunks(self, chunks: list[ChunkArtifact]) -> None:
        for reasoning in self._inputs.reasoning_artifacts:
            chunks.extend(_reasoning_to_chunks(reasoning, self._manifests))


class DefaultChunkStage(ChunkStage):
    def build_chunks(
        self,
        *,
        page_manifests: list[PageManifest],
        regions: list[Region],
        tables: list[TableArtifact],
        figures: list[FigureArtifact],
        reasoning_artifacts: list[ReasoningArtifact] | None = None,
    ) -> list[ChunkArtifact]:
        return build_chunk_artifacts(
            page_manifests,
            regions,
            tables,
            figures,
            reasoning_artifacts or [],
        )

    def write_bundle(
        self,
        *,
        artifact_root: Path,
        liteparse_pages: list[dict[str, Any]],
        pymupdf_pages: list[dict[str, Any]],
        page_manifests: list[PageManifest],
        regions: list[Region],
        table_fragments: list[TableArtifact],
        merged_tables: list[TableArtifact],
        figures: list[FigureArtifact],
        chunks: list[ChunkArtifact],
        reasoning_artifacts: list[ReasoningArtifact] | None = None,
    ) -> None:
        write_artifact_bundle(
            artifact_root=artifact_root,
            liteparse_pages=liteparse_pages,
            pymupdf_pages=pymupdf_pages,
            page_manifests=page_manifests,
            regions=regions,
            table_fragments=table_fragments,
            merged_tables=merged_tables,
            figures=figures,
            chunks=chunks,
            reasoning_artifacts=reasoning_artifacts or [],
        )

    def to_llama_docs(
        self,
        *,
        chunk_artifacts: list[ChunkArtifact],
        document_metadata: dict[str, Any],
        source_file: str,
        artifact_root: Path,
    ) -> tuple[list[LlamaDocument], list[dict[str, Any]]]:
        return artifact_chunks_to_llama_docs(
            chunk_artifacts=chunk_artifacts,
            document_metadata=document_metadata,
            source_file=source_file,
            artifact_root=artifact_root,
        )


def build_chunk_artifacts(
    page_manifests: list[PageManifest],
    regions: list[Region],
    tables: list[TableArtifact],
    figures: list[FigureArtifact],
    reasoning_artifacts: list[ReasoningArtifact] | None = None,
) -> list[ChunkArtifact]:
    inputs = ChunkBuildInputs(
        page_manifests=page_manifests,
        regions=regions,
        tables=tables,
        figures=figures,
        reasoning_artifacts=reasoning_artifacts or [],
    )
    return ChunkArtifactAssembler(inputs).build()


def artifact_chunks_to_llama_docs(
    chunk_artifacts: list[ChunkArtifact],
    document_metadata: dict[str, Any],
    source_file: str,
    artifact_root: Path,
) -> tuple[list[LlamaDocument], list[dict[str, Any]]]:
    docs: list[LlamaDocument] = []
    units: list[dict[str, Any]] = []

    for artifact in chunk_artifacts:
        text = artifact.text.strip()
        if not text:
            continue

        first_page = artifact.page_nums[0] if artifact.page_nums else 1
        metadata = {
            **document_metadata,
            **artifact.metadata,
            "source_file": source_file,
            "page_num": first_page,
            "page_nums": artifact.page_nums,
            "chunk_id": artifact.chunk_id,
            "chunk_type": artifact.chunk_type,
            "source_artifact_type": artifact.source_artifact_type,
            "source_artifact_id": artifact.source_artifact_id,
            "asset_refs": artifact.asset_refs,
            "artifact_bundle_path": str(artifact_root),
            "parser_name": "rag_parser",
            "parser_version": settings.PDF_LAYOUT_PARSER_VERSION,
            "table_detected": artifact.chunk_type
            in {"full_table", "table_segment", "table_summary_text"},
            "chart_detected": artifact.chunk_type
            in {
                "figure_artifact",
                "chart_context",
                "visual_proxy_text",
                "chart_data_points",
            },
            "contains_numeric_data": has_numeric_data(text),
        }
        metadata["numeric_density"] = max(
            float(metadata.get("numeric_density", 0.0) or 0.0),
            numeric_density(text),
        )

        docs.append(LlamaDocument(text=text, metadata=metadata))
        units.append(
            {
                "id": artifact.chunk_id,
                "document_id": document_metadata.get("document_id"),
                "unit_type": artifact.chunk_type,
                "page_num": first_page,
                "page_nums": artifact.page_nums,
                "raw_text": text,
                "table_detected": metadata["table_detected"],
                "chart_detected": metadata["chart_detected"],
                "contains_numeric_data": metadata["contains_numeric_data"],
                "chunk_type": artifact.chunk_type,
                "source_artifact_type": artifact.source_artifact_type,
                "source_artifact_id": artifact.source_artifact_id,
                "layout_confidence": metadata.get("layout_confidence"),
                "complexity_score": metadata.get("complexity_score"),
                "artifact_bundle_path": str(artifact_root),
                "layout_engine": metadata.get("layout_engine"),
                "pdf_repair_attempted": metadata.get("pdf_repair_attempted", False),
                "pdf_repair_method": metadata.get("pdf_repair_method"),
                "pdf_repair_success": metadata.get("pdf_repair_success", False),
                "llm_enriched": metadata.get("llm_enriched", False),
                "degraded_stages": metadata.get("degraded_stages", []),
                "page_parse_degraded": metadata.get("page_parse_degraded", False),
            }
        )

    return docs, units


def write_artifact_bundle(
    artifact_root: Path,
    liteparse_pages: list[dict[str, Any]],
    pymupdf_pages: list[dict[str, Any]],
    page_manifests: list[PageManifest],
    regions: list[Region],
    table_fragments: list[TableArtifact],
    merged_tables: list[TableArtifact],
    figures: list[FigureArtifact],
    chunks: list[ChunkArtifact],
    reasoning_artifacts: list[ReasoningArtifact] | None = None,
) -> None:
    raw_dir = artifact_root / "raw"
    dump_json(raw_dir / "liteparse_pages.json", liteparse_pages)
    dump_json(raw_dir / "pymupdf_pages.json", pymupdf_pages)
    dump_json(
        artifact_root / "page_manifests.json", [asdict(item) for item in page_manifests]
    )
    dump_json(artifact_root / "regions.json", [asdict(item) for item in regions])
    dump_json(
        artifact_root / "table_fragments.json",
        [asdict(item) for item in table_fragments],
    )
    dump_json(
        artifact_root / "merged_tables.json", [asdict(item) for item in merged_tables]
    )
    dump_json(
        artifact_root / "figure_artifacts.json", [asdict(item) for item in figures]
    )
    dump_json(artifact_root / "chunk_artifacts.json", [asdict(item) for item in chunks])
    if reasoning_artifacts:
        dump_json(
            artifact_root / "reasoning_artifacts.json",
            [asdict(item) for item in reasoning_artifacts],
        )


def split_body_text(text: str) -> list[str]:
    max_tokens = max(1, int(settings.BODY_TEXT_CHUNK_MAX_TOKENS))
    overlap_tokens = max(0, int(settings.BODY_TEXT_CHUNK_OVERLAP_TOKENS))
    if overlap_tokens >= max_tokens:
        overlap_tokens = max_tokens - 1

    if count_tokens(text, encoding_name=settings.TOKEN_BUDGET_ENCODING) <= max_tokens:
        return [text]

    try:
        sentence_chunker = (
            PDFPipelineRegistry().chunker_manager.create_sentence_chunker(
                tokenizer=settings.TOKEN_BUDGET_ENCODING,
                chunk_size=max_tokens,
                chunk_overlap=overlap_tokens,
            )
        )
        if sentence_chunker is None:
            return _token_window_split(
                text=text,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )

        chunks = _normalize_chunk_texts(sentence_chunker.chunk(text))
        return chunks or _token_window_split(
            text=text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
    except Exception:
        logger.exception("SentenceChunker failed; falling back to token windowing")
        return _token_window_split(
            text=text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )


def _token_window_split(
    *,
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    if max_tokens <= 0:
        return [text]
    encoding = get_token_encoding(encoding_name=settings.TOKEN_BUDGET_ENCODING)
    token_ids = encoding.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]

    step = max(1, max_tokens - overlap_tokens)
    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        end = min(len(token_ids), start + max_tokens)
        chunk_text = encoding.decode(token_ids[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(token_ids):
            break
        start += step

    return chunks or [text]


def _table_to_chunks(
    table: TableArtifact,
    manifests: dict[int, PageManifest],
) -> list[ChunkArtifact]:
    rows = table.json_table or []
    markdown = table_to_markdown(rows)
    normalized_text = table.normalized_table_text or markdown or table.caption_text
    if not normalized_text.strip():
        return []

    primary_page = table.page_nums[0] if table.page_nums else 1
    manifest = manifests.get(primary_page)
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


def _reasoning_to_chunks(
    reasoning: ReasoningArtifact,
    manifests: dict[int, PageManifest],
) -> list[ChunkArtifact]:
    """Convert a ReasoningArtifact into one or more indexable chunks."""
    if not reasoning.text.strip():
        return []

    primary_page = reasoning.page_nums[0] if reasoning.page_nums else 1
    manifest = manifests.get(primary_page)

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

    return [
        ChunkArtifact(
            chunk_id=str(uuid.uuid4()),
            chunk_type=chunk_type,
            source_artifact_type="reasoning",
            source_artifact_id=reasoning.reasoning_id,
            page_nums=reasoning.page_nums,
            text=reasoning.text,
            metadata=metadata,
        )
    ]


def _figure_to_chunks(
    figure: FigureArtifact,
    manifests: dict[int, PageManifest],
) -> list[ChunkArtifact]:
    manifest = manifests.get(figure.page_num)
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
    metadata["llm_enriched"] = figure.llm_enriched
    metadata["llm_enrichment_confidence"] = figure.llm_enrichment_confidence
    metadata["chart_type"] = figure.chart_type
    metadata["chart_title"] = figure.chart_title
    metadata["x_axis_label"] = figure.x_axis_label
    metadata["y_axis_label"] = figure.y_axis_label
    metadata["x_categories"] = figure.x_categories
    metadata["series"] = figure.series
    metadata["approx_datapoints"] = figure.approx_datapoints
    metadata["trend_summary"] = figure.trend_summary
    metadata["key_chart_facts"] = figure.key_chart_facts
    metadata["numeric_extraction_confidence"] = figure.numeric_extraction_confidence
    metadata["chart_parse_status"] = figure.chart_parse_status
    metadata["llm_caption_model"] = figure.llm_caption_model
    metadata["llm_caption_version"] = figure.llm_caption_version
    metadata["llm_caption_prompt_version"] = figure.llm_caption_prompt_version
    metadata["llm_caption_status"] = figure.llm_caption_status
    metadata["llm_caption_error"] = figure.llm_caption_error

    preferred_assets = (
        [figure.crop_path] if figure.crop_path else [figure.page_screenshot_path]
    )
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


def _classify_slide_purpose(manifest: "PageManifest | None") -> str:
    """Classify a slide's primary purpose for retrieval priority boosting.

    Returns one of: overview_stats | chart_analysis | table_data | narrative
    """
    if manifest is None:
        return "narrative"

    page_text_lower = (manifest.full_page_text or "").lower()

    if any(kw in page_text_lower for kw in _OVERVIEW_HEADING_KEYWORDS):
        return "overview_stats"

    if manifest.page_class == "visual_heavy_page":
        return "chart_analysis"

    if manifest.page_class == "table_heavy_page":
        return "table_data"

    # Quick-facts slides: short text with dense numeric-labeled pairs
    # Heuristic: high numeric density + short page text = stats slide
    page_text = manifest.full_page_text or ""
    if len(page_text) < 800 and numeric_density(page_text) > 0.08:
        return "overview_stats"

    return "narrative"


_VISIBLE_PAGE_PATTERN = re.compile(
    r"(?:^|\s)(\d{1,3})(?:\s*$)",  # standalone number at end of text
    re.MULTILINE,
)
_PAGE_LABEL_PATTERN = re.compile(r"(?:page|slide|pg\.?)\s*(\d{1,3})", re.IGNORECASE)


def _extract_visible_page_num(manifest: "PageManifest | None") -> int | None:
    """Extract the visible slide/page number from page text, if present.

    Investor decks typically show a page number in the footer (e.g. "14" or "Page 14").
    Returns the extracted integer, or None if not found or ambiguous.
    """
    if manifest is None:
        return None
    text = (manifest.full_page_text or "").strip()
    if not text:
        return None

    # Try "Page N" / "Slide N" pattern first (unambiguous)
    for match in _PAGE_LABEL_PATTERN.finditer(text):
        num = int(match.group(1))
        if 1 <= num <= 300:
            return num

    # Fall back to standalone number at the very end of the page (footer)
    tail = text[-120:]
    for match in _VISIBLE_PAGE_PATTERN.finditer(tail):
        num = int(match.group(1))
        if 1 <= num <= 300:
            return num

    return None


def _regions_by_page(regions: list[Region]) -> dict[int, list[Region]]:
    grouped: dict[int, list[Region]] = {}
    for region in regions:
        grouped.setdefault(region.page_num, []).append(region)
    for region_list in grouped.values():
        region_list.sort(key=lambda region: region.reading_order)
    return grouped


def _group_regions_by_section(regions: list[Region]) -> dict[str, list[Region]]:
    grouped: dict[str, list[Region]] = defaultdict(list)
    for region in sorted(regions, key=lambda item: item.reading_order):
        section = region.section_path or "Document"
        grouped[section].append(region)
    return dict(grouped)


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
    if figure.numeric_extraction_confidence is not None:
        lines.append(
            f"Numeric extraction confidence: {figure.numeric_extraction_confidence:.2f}"
        )
    lines.append("Note: numeric values may be approximate.")
    return "\n".join(line for line in lines if line.strip())
