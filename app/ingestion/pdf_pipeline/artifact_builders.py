from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import mimetypes
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

from PIL import Image, ImageStat

import pymupdf as fitz
from llama_index.core import Settings as LlamaSettings
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)

from app.core.config import settings
from app.core.prompts import (
    CHART_REASONING_PROMPT,
    PAGE_REASONING_PROMPT,
    PAGE_SCREENSHOT_PROMPT,
    REASONING_PROMPT_VERSION,
    TABLE_REASONING_PROMPT,
    build_artifact_enrichment_prompt,
    build_chart_caption_prompt,
)
from app.indexing.vector_store import vector_store_manager
from app.ingestion.pdf_pipeline.contracts import ArtifactResult, ArtifactStage
from app.ingestion.pdf_pipeline.helpers import (
    extract_units,
    header_signature,
    numeric_density,
    normalize_table_rows,
    normalize_whitespace,
    parse_table_like_text,
    table_rows_to_html,
    to_float_bbox,
    union_bbox,
)
from app.ingestion.pdf_pipeline.models import (
    FigureArtifact,
    PageManifest,
    ReasoningArtifact,
    Region,
    TableArtifact,
)

logger = logging.getLogger(__name__)

_Job = TypeVar("_Job")
_Result = TypeVar("_Result")


def _llm_available_for_pipeline() -> bool:
    try:
        vector_store_manager.configure_llama_settings()
        return True
    except Exception as exc:
        logger.warning("Skipping LLM artifact step: %s", exc)
        return False


@dataclass(frozen=True, slots=True)
class ParallelExecutionConfig:
    max_parallel_requests: int = 4


class ParallelExecutor:
    def __init__(self, config: ParallelExecutionConfig | None = None) -> None:
        self._config = config or ParallelExecutionConfig()

    def worker_count(self, job_count: int, max_workers: int | None = None) -> int:
        limit = max_workers or self._config.max_parallel_requests
        return max(1, min(job_count, limit))

    def run(
        self,
        jobs: list[_Job],
        runner: Callable[[_Job], _Result],
        *,
        max_workers: int | None = None,
        timeout_seconds: float | None = None,
    ) -> list[tuple[_Job, _Result | None, Exception | None]]:
        if not jobs:
            return []

        workers = self.worker_count(len(jobs), max_workers=max_workers)

        def _wrapped(job: _Job) -> tuple[_Job, _Result | None, Exception | None]:
            try:
                return (job, runner(job), None)
            except Exception as exc:
                return (job, None, exc)

        if workers == 1 and timeout_seconds is None:
            return [_wrapped(job) for job in jobs]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(runner, job) for job in jobs]
            results: list[tuple[_Job, _Result | None, Exception | None]] = []
            for job, future in zip(jobs, futures):
                try:
                    result = future.result(timeout=timeout_seconds)
                    results.append((job, result, None))
                except FuturesTimeoutError:
                    future.cancel()
                    timeout_error = TimeoutError(
                        f"job exceeded timeout ({timeout_seconds}s)"
                    )
                    results.append((job, None, timeout_error))
                except Exception as exc:
                    results.append((job, None, exc))
            return results


_PARALLEL_EXECUTOR = ParallelExecutor()


def _parallel_worker_count(
    job_count: int,
    *,
    max_workers: int | None = None,
) -> int:
    return _PARALLEL_EXECUTOR.worker_count(job_count, max_workers=max_workers)


def _run_parallel_jobs(
    jobs: list[_Job],
    runner: Callable[[_Job], _Result],
    *,
    max_workers: int | None = None,
    timeout_seconds: float | None = None,
) -> list[tuple[_Job, _Result | None, Exception | None]]:
    return _PARALLEL_EXECUTOR.run(
        jobs,
        runner,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
    )


class DefaultArtifactStage(ArtifactStage):
    def build(
        self,
        *,
        pdf_doc: fitz.Document,
        page_manifests: list[PageManifest],
        pymupdf_pages: list[dict[str, Any]],
        regions: list[Region],
        figure_dir: Path,
    ) -> ArtifactResult:
        table_fragments = build_table_artifacts(pymupdf_pages, regions)
        merged_tables = merge_table_artifacts(table_fragments)
        figures = build_figure_artifacts(
            pdf_doc=pdf_doc,
            page_manifests=page_manifests,
            pymupdf_pages=pymupdf_pages,
            regions=regions,
            figure_dir=figure_dir,
        )
        analyze_chart_artifacts(page_manifests, figures)
        maybe_llm_enrich_artifacts(page_manifests, merged_tables, figures)
        analyze_page_screenshots(page_manifests)
        reasoning_artifacts = run_reasoning_enrichment(
            page_manifests,
            merged_tables,
            figures,
        )
        return ArtifactResult(
            table_fragments=table_fragments,
            merged_tables=merged_tables,
            figures=figures,
            reasoning_artifacts=reasoning_artifacts,
        )


def build_table_artifacts(
    pymupdf_pages: list[dict[str, Any]],
    regions: list[Region],
) -> list[TableArtifact]:
    artifacts: list[TableArtifact] = []
    by_page = _regions_by_page(regions)

    for page in pymupdf_pages:
        page_num = int(page.get("page_num", 1))
        page_regions = by_page.get(page_num, [])
        candidates = list(page.get("table_candidates") or [])

        if not candidates:
            table_regions = [
                region
                for region in page_regions
                if region.region_type == "table_region"
            ]
            if table_regions:
                merged_rows = parse_table_like_text(
                    "\n".join(region.text for region in table_regions)
                )
                if len(merged_rows) >= 2:
                    candidates.append(
                        {
                            "candidate_id": f"text_regions_page_{page_num}",
                            "bbox": _union_region_bbox(table_regions),
                            "rows": merged_rows,
                        }
                    )

        for candidate in candidates:
            rows = _normalize_rows(candidate.get("rows") or [])
            if len(rows) < 2:
                continue

            bbox = to_float_bbox(candidate.get("bbox"))
            caption = _nearest_caption(page_regions, bbox)
            section_path = _section_for_bbox(page_regions, bbox)
            footnotes = _footnotes_for_bbox(page_regions, bbox)
            normalized_table_text = normalize_table_rows(rows)
            header = [normalize_whitespace(cell) for cell in rows[0]] if rows else []

            artifacts.append(
                TableArtifact(
                    table_id=str(uuid.uuid4()),
                    page_nums=[page_num],
                    bbox_list=[bbox],
                    caption_text=caption,
                    section_path=section_path,
                    html_table=table_rows_to_html(rows),
                    json_table=rows,
                    normalized_table_text=normalized_table_text,
                    header_rows=header,
                    units=extract_units(f"{caption} {normalized_table_text}"),
                    footnotes=footnotes,
                    continuation_flag=False,
                    confidence=0.86 if caption else 0.74,
                )
            )

    return artifacts


def merge_table_artifacts(tables: list[TableArtifact]) -> list[TableArtifact]:
    if not settings.ENABLE_MULTIPAGE_TABLE_MERGE or not tables:
        return tables

    merged: list[TableArtifact] = []
    ordered = sorted(tables, key=lambda item: (min(item.page_nums), item.table_id))

    for table in ordered:
        if not merged:
            merged.append(table)
            continue

        prev = merged[-1]
        if not _should_merge_tables(prev, table):
            merged.append(table)
            continue

        prev.page_nums = sorted(set(prev.page_nums + table.page_nums))
        prev.bbox_list.extend(table.bbox_list)

        prev_rows = prev.json_table or []
        next_rows = table.json_table or []
        if prev_rows and next_rows and _same_headers(prev_rows[0], next_rows[0]):
            prev_rows.extend(next_rows[1:])
        else:
            prev_rows.extend(next_rows)
        prev.json_table = prev_rows

        prev.normalized_table_text = normalize_table_rows(prev_rows)
        prev.html_table = table_rows_to_html(prev_rows)
        prev.units = sorted(set(prev.units + table.units))
        prev.footnotes = _dedupe_strings(prev.footnotes + table.footnotes)
        prev.continuation_flag = True
        prev.confidence = max(prev.confidence, table.confidence)

    return merged


def build_figure_artifacts(
    pdf_doc: fitz.Document,
    page_manifests: list[PageManifest],
    pymupdf_pages: list[dict[str, Any]],
    regions: list[Region],
    figure_dir: Path,
) -> list[FigureArtifact]:
    artifacts: list[FigureArtifact] = []
    by_page = _regions_by_page(regions)
    manifests = {manifest.page_num: manifest for manifest in page_manifests}

    for page in pymupdf_pages:
        page_num = int(page.get("page_num", 1))
        page_regions = by_page.get(page_num, [])
        manifest = manifests.get(page_num)

        candidates: list[dict[str, Any]] = []
        for image in page.get("image_refs") or []:
            bbox = image.get("bbox")
            if bbox:
                candidates.append({"bbox": to_float_bbox(bbox), "kind": "image"})

        vector_count = int(page.get("vector_count") or 0)
        vector_union = union_bbox(page.get("vector_bboxes") or [])
        if vector_count >= 20 and vector_union:
            candidates.append({"bbox": vector_union, "kind": "vector_dense"})

        if not candidates:
            for region in page_regions:
                if region.region_type == "figure_region":
                    candidates.append({"bbox": region.bbox, "kind": "text_figure_hint"})

        deduped = _dedupe_candidates(candidates)
        if not deduped:
            continue

        for idx, candidate in enumerate(deduped, start=1):
            bbox = candidate["bbox"]
            caption = _nearest_caption(page_regions, bbox)
            nearby = _nearest_explanatory_text(page_regions, bbox)
            section_path = _section_for_bbox(page_regions, bbox)
            footnotes = _footnotes_for_bbox(page_regions, bbox)
            figure_type = _classify_figure(caption, nearby, candidate["kind"])
            crop_path = _save_figure_crop(
                pdf_doc[page_num - 1], bbox, figure_dir, page_num, idx
            )

            if crop_path:
                if not validate_image_crop(crop_path):
                    logger.debug(
                        "Discarding wasteful or decorative figure crop: %s", crop_path
                    )
                    try:
                        Path(crop_path).unlink(missing_ok=True)
                    except OSError:
                        pass

                    # If the region had no caption or nearby text, and the image is wasteful,
                    # discard the candidate entirely.
                    if not caption and not nearby:
                        continue

                    crop_path = ""

            artifacts.append(
                FigureArtifact(
                    figure_id=str(uuid.uuid4()),
                    page_num=page_num,
                    bbox=bbox,
                    figure_type=figure_type,
                    caption_text=caption,
                    nearby_text=nearby,
                    section_path=section_path,
                    crop_path=crop_path,
                    page_screenshot_path=manifest.screenshot_path if manifest else "",
                    visual_proxy_text=_build_visual_proxy_text(
                        page_num=page_num,
                        section_path=section_path,
                        figure_type=figure_type,
                        caption=caption,
                        nearby_text=nearby,
                    ),
                    footnotes=footnotes,
                    confidence=0.83 if caption else 0.72,
                )
            )

    return artifacts


def analyze_chart_artifacts(
    page_manifests: list[PageManifest],
    figures: list[FigureArtifact],
) -> None:
    if not figures:
        return

    manifest_by_page = {manifest.page_num: manifest for manifest in page_manifests}
    chart_candidates = [
        figure
        for figure in figures
        if _is_chart_candidate(figure.figure_type, figure.caption_text)
    ]
    if not chart_candidates:
        return

    for figure in chart_candidates:
        figure.chart_parse_status = "partial"

    if not settings.ENABLE_MULTIMODAL_CAPTIONING:
        for figure in chart_candidates:
            figure.llm_caption_status = "skipped"
            _apply_heuristic_chart_fields(figure)
        return
    if not _llm_available_for_pipeline():
        for figure in chart_candidates:
            figure.llm_caption_status = "skipped"
            figure.llm_caption_error = "google_api_key_unavailable"
            _apply_heuristic_chart_fields(figure)
        return

    max_pages = max(1, int(settings.LLM_CAPTION_MAX_PAGES))
    max_artifacts = max(1, int(settings.LLM_CAPTION_MAX_ARTIFACTS_PER_PAGE))
    eligible_pages = [
        manifest.page_num
        for manifest in sorted(
            page_manifests,
            key=lambda item: item.complexity_score,
            reverse=True,
        )
        if manifest.page_class in {"visual_heavy_page", "hard_page", "table_heavy_page"}
    ][:max_pages]
    eligible_page_set = set(eligible_pages)

    figures_by_page: dict[int, list[FigureArtifact]] = {}
    for figure in chart_candidates:
        figures_by_page.setdefault(figure.page_num, []).append(figure)

    llm_jobs: list[tuple[FigureArtifact, str]] = []

    for page_num, page_figures in figures_by_page.items():
        if page_num not in eligible_page_set:
            for figure in page_figures:
                figure.llm_caption_status = "skipped"
                _apply_heuristic_chart_fields(figure)
            continue

        for figure in page_figures[:max_artifacts]:
            manifest = manifest_by_page.get(figure.page_num)
            prompt = build_chart_caption_prompt(
                page_num=figure.page_num,
                page_class=manifest.page_class if manifest else "unknown",
                caption_text=figure.caption_text,
                nearby_text=figure.nearby_text,
                current_proxy=_chart_summary_text(figure),
            )
            llm_jobs.append((figure, prompt))

        for figure in page_figures[max_artifacts:]:
            figure.llm_caption_status = "skipped"
            _apply_heuristic_chart_fields(figure)

    def _run_chart_job(job: tuple[FigureArtifact, str]) -> str:
        figure, prompt = job
        return _run_chart_inference(prompt, figure.crop_path)

    chart_results = _run_parallel_jobs(
        llm_jobs,
        _run_chart_job,
        max_workers=_parallel_worker_count(len(llm_jobs)),
        timeout_seconds=float(settings.LLM_CAPTION_TIMEOUT_SECONDS),
    )

    for job, response_text, error in chart_results:
        figure, _prompt = job
        if error is not None:
            logger.error(
                "Chart captioning failed for figure %s",
                figure.figure_id,
                exc_info=error,
            )
            figure.llm_caption_status = "failed"
            figure.llm_caption_error = str(error)[:400]
            _apply_heuristic_chart_fields(figure)
            continue

        parsed = _parse_chart_json_response(response_text or "")
        if not parsed:
            figure.llm_caption_status = "failed"
            figure.llm_caption_error = "invalid_json_response"
            _apply_heuristic_chart_fields(figure)
            continue

        _apply_llm_chart_fields(figure, parsed)
        if not figure.approx_datapoints and not figure.key_chart_facts:
            _apply_heuristic_chart_fields(figure)
        figure.llm_caption_status = "success"
        figure.llm_caption_model = settings.LLM_MODEL
        figure.llm_caption_version = settings.PDF_LAYOUT_PARSER_VERSION
        figure.llm_caption_prompt_version = "chart_caption_v1"
        figure.llm_caption_error = None
        figure.llm_enriched = True
        confidence = figure.numeric_extraction_confidence or 0.75
        figure.llm_enrichment_confidence = max(
            figure.llm_enrichment_confidence or 0.0, confidence
        )
        if figure.approx_datapoints or figure.key_chart_facts or figure.trend_summary:
            figure.chart_parse_status = "success"
        else:
            figure.chart_parse_status = "partial"

        figure.visual_proxy_text = _build_visual_proxy_text(
            page_num=figure.page_num,
            section_path=figure.section_path,
            figure_type=figure.chart_type or figure.figure_type,
            caption=figure.caption_text or figure.chart_title,
            nearby_text=_chart_summary_text(figure),
        )


def maybe_llm_enrich_artifacts(
    page_manifests: list[PageManifest],
    tables: list[TableArtifact],
    figures: list[FigureArtifact],
) -> None:
    if not settings.ENABLE_LLM_ARTIFACT_ENRICHMENT:
        return
    if not _llm_available_for_pipeline():
        return

    max_pages = max(1, int(settings.LLM_ARTIFACT_ENRICHMENT_MAX_PAGES))
    eligible_pages = [
        manifest.page_num
        for manifest in sorted(
            page_manifests,
            key=lambda item: item.complexity_score,
            reverse=True,
        )
        if manifest.page_class != "simple_text_page"
    ][:max_pages]

    if not eligible_pages:
        return

    table_map: dict[int, list[TableArtifact]] = {}
    for table in tables:
        for page_num in table.page_nums:
            table_map.setdefault(page_num, []).append(table)

    figure_map: dict[int, list[FigureArtifact]] = {}
    for figure in figures:
        figure_map.setdefault(figure.page_num, []).append(figure)

    enrichment_jobs: list[
        tuple[int, list[TableArtifact], list[FigureArtifact], str]
    ] = []
    for page_num in eligible_pages:
        page_tables = table_map.get(page_num, [])[:2]
        page_figures = figure_map.get(page_num, [])[:2]
        if not page_tables and not page_figures:
            continue

        table_summaries = [
            (table.caption_text, table.normalized_table_text) for table in page_tables
        ]
        figure_summaries = [
            (figure.caption_text, figure.nearby_text) for figure in page_figures
        ]
        prompt = build_artifact_enrichment_prompt(
            page_num=page_num,
            table_summaries=table_summaries,
            figure_summaries=figure_summaries,
        )
        enrichment_jobs.append((page_num, page_tables, page_figures, prompt))

    def _run_enrichment_job(
        job: tuple[int, list[TableArtifact], list[FigureArtifact], str],
    ) -> str:
        _page_num, _page_tables, _page_figures, prompt = job
        response = LlamaSettings.llm.complete(prompt)
        return normalize_whitespace(str(response))

    for job, enrichment, error in _run_parallel_jobs(
        enrichment_jobs,
        _run_enrichment_job,
        max_workers=_parallel_worker_count(len(enrichment_jobs)),
        timeout_seconds=float(settings.LLM_CAPTION_TIMEOUT_SECONDS),
    ):
        page_num, page_tables, page_figures, _prompt = job
        if error is not None:
            logger.error("LLM enrichment failed for page %s", page_num, exc_info=error)
            continue

        if not enrichment:
            continue

        for table in page_tables:
            table.normalized_table_text = f"{table.normalized_table_text}\nLLM summary: {enrichment[:600]}".strip()
            table.llm_enriched = True
            table.llm_enrichment_confidence = min(1.0, table.confidence + 0.07)

        for figure in page_figures:
            if figure.llm_caption_status == "success":
                continue
            figure.visual_proxy_text = (
                f"{figure.visual_proxy_text}\nLLM summary: {enrichment[:600]}".strip()
            )
            figure.llm_enriched = True
            figure.llm_enrichment_confidence = min(1.0, figure.confidence + 0.07)


def _regions_by_page(regions: list[Region]) -> dict[int, list[Region]]:
    grouped: dict[int, list[Region]] = {}
    for region in regions:
        grouped.setdefault(region.page_num, []).append(region)
    for region_list in grouped.values():
        region_list.sort(key=lambda region: region.reading_order)
    return grouped


def _union_region_bbox(regions: list[Region]) -> list[float]:
    if not regions:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = min(region.bbox[0] for region in regions)
    y0 = min(region.bbox[1] for region in regions)
    x1 = max(region.bbox[2] for region in regions)
    y1 = max(region.bbox[3] for region in regions)
    return [x0, y0, x1, y1]


def _normalize_rows(rows: list[list[Any]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for row in rows:
        if not row:
            continue
        normalized_row = [normalize_whitespace(str(cell or "")) for cell in row]
        if any(cell for cell in normalized_row):
            normalized.append(normalized_row)
    return normalized


def _nearest_caption(regions: list[Region], bbox: list[float]) -> str:
    candidates = [region for region in regions if region.region_type == "caption"]
    if not candidates:
        return ""

    anchor = bbox[1]
    best = min(candidates, key=lambda region: abs(region.bbox[1] - anchor))
    return best.text


def _section_for_bbox(regions: list[Region], bbox: list[float]) -> str:
    anchor = bbox[1]
    best_heading: Region | None = None
    for region in regions:
        if region.region_type not in {"title", "section_heading", "subsection_heading"}:
            continue
        if region.bbox[1] > anchor:
            continue
        if best_heading is None or region.bbox[1] > best_heading.bbox[1]:
            best_heading = region

    if best_heading:
        return best_heading.section_path
    return "Document"


def _footnotes_for_bbox(regions: list[Region], bbox: list[float]) -> list[str]:
    notes: list[str] = []
    for region in regions:
        if region.region_type != "footnote":
            continue
        if region.bbox[1] >= bbox[3] - 20:
            notes.append(region.text)
    return notes[:3]


def _nearest_explanatory_text(regions: list[Region], bbox: list[float]) -> str:
    candidates: list[tuple[float, str]] = []
    for region in regions:
        if region.region_type not in {"body_text", "section_heading"}:
            continue
        distance = abs(region.bbox[1] - bbox[1])
        if distance <= 260:
            candidates.append((distance, region.text))

    candidates.sort(key=lambda item: item[0])
    return "\n".join(text for _, text in candidates[:2])


def _classify_figure(caption: str, nearby_text: str, candidate_type: str) -> str:
    haystack = f"{caption} {nearby_text}".lower()
    if "infographic" in haystack:
        return "infographic"
    if "diagram" in haystack:
        return "diagram"
    if any(token in haystack for token in ("chart", "graph", "plot", "trend")):
        return "chart"
    if candidate_type == "vector_dense":
        return "diagram"
    if candidate_type == "image":
        return "photo"
    return "unknown"


def _build_visual_proxy_text(
    page_num: int,
    section_path: str,
    figure_type: str,
    caption: str,
    nearby_text: str,
) -> str:
    lines = [
        f"Page: {page_num}",
        f"Section: {section_path}",
        f"Figure type: {figure_type}",
    ]
    if caption:
        lines.append(f"Caption: {caption}")
    if nearby_text:
        lines.append(f"Nearby text: {nearby_text}")
    return "\n".join(lines)


def _save_figure_crop(
    page: fitz.Page,
    bbox: list[float],
    figure_dir: Path,
    page_num: int,
    index: int,
) -> str:
    figure_dir.mkdir(parents=True, exist_ok=True)
    page_rect = page.rect
    clip = fitz.Rect(*bbox) & page_rect
    if clip.is_empty:
        return ""

    out_path = figure_dir / f"page_{page_num}_figure_{index}.png"
    try:
        page.get_pixmap(clip=clip, dpi=140).save(str(out_path))
    except Exception:
        logger.exception("Failed to save figure crop for page %s", page_num)
        return ""
    return str(out_path)


def validate_image_crop(image_path: str) -> bool:
    """
    Heuristically validate if a cropped image contains meaningful information.
    Rejects:
    - Extremely small sizes (scanning artifacts, icons, thin lines)
    - Extreme aspect ratios (horizontal/vertical separator lines)
    - Low variance/entropy (solid color blocks or completely blank images)
    """
    if not image_path or not Path(image_path).exists():
        return False

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 30 or height < 30:
                return False

            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio > 12.0 or aspect_ratio < (1.0 / 12.0):
                return False

            stat = ImageStat.Stat(img.convert("L"))
            variance = stat.var[0] if stat.var else 0

            # Very low variance means the image is mostly a solid block of color
            if variance < 2.0:
                return False

            return True
    except Exception as e:
        logger.warning("Error validating image crop %s: %s", image_path, e)
        return False


def _same_headers(left: list[str], right: list[str]) -> bool:
    return header_signature(left) == header_signature(right)


def _should_merge_tables(left: TableArtifact, right: TableArtifact) -> bool:
    left_page = max(left.page_nums) if left.page_nums else 0
    right_page = min(right.page_nums) if right.page_nums else 0
    if right_page != left_page + 1:
        return False

    same_header = bool(left.header_rows) and _same_headers(
        left.header_rows, right.header_rows
    )
    same_caption = (
        normalize_whitespace(left.caption_text).lower()
        and normalize_whitespace(left.caption_text).lower()
        == normalize_whitespace(right.caption_text).lower()
    )
    aligned = _rough_x_alignment(left.bbox_list[-1], right.bbox_list[0])
    return aligned and (same_header or same_caption)


def _rough_x_alignment(left_bbox: list[float], right_bbox: list[float]) -> bool:
    return (
        abs(left_bbox[0] - right_bbox[0]) <= 28
        and abs(left_bbox[2] - right_bbox[2]) <= 28
    )


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = normalize_whitespace(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for candidate in candidates:
        bbox = to_float_bbox(candidate.get("bbox"))
        if _bbox_area(bbox) <= 1.0:
            continue
        if any(_bbox_iou(bbox, current["bbox"]) > 0.8 for current in accepted):
            continue
        accepted.append({"bbox": bbox, "kind": candidate.get("kind", "unknown")})
    return accepted


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_iou(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])

    inter = _bbox_area([x0, y0, x1, y1])
    if inter <= 0:
        return 0.0

    union = _bbox_area(left) + _bbox_area(right) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _is_chart_candidate(figure_type: str, caption_text: str) -> bool:
    if figure_type in {"chart", "diagram", "infographic"}:
        return True
    lowered = normalize_whitespace(caption_text).lower()
    return any(
        token in lowered
        for token in ("chart", "graph", "bar", "line", "trend", "histogram")
    )


def _run_chart_inference(prompt: str, image_path: str) -> str:
    multimodal_text = _run_google_multimodal_inference(prompt, image_path)
    if multimodal_text:
        return multimodal_text

    response = LlamaSettings.llm.complete(prompt)
    return str(response)


def _run_google_multimodal_inference(prompt: str, image_path: str) -> str | None:
    if (
        not image_path
        or settings.is_google_api_key_placeholder
        or not Path(image_path).exists()
    ):
        return None

    try:
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        response = LlamaSettings.llm.chat(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    blocks=[
                        TextBlock(text=prompt),
                        ImageBlock(path=image_path, image_mimetype=mime_type),
                    ],
                )
            ]
        )

        message = getattr(response, "message", None)
        if message is not None:
            parts: list[str] = []
            for block in getattr(message, "blocks", None) or []:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text).strip())
            if parts:
                return "\n".join(part for part in parts if part)

            content = getattr(message, "content", None)
            if content:
                return str(content)

        response_text = str(response).strip()
        if response_text:
            return response_text
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        error_text = str(exc).lower()
        transient_provider_error = status_code in {429, 500, 502, 503, 504} or any(
            marker in error_text
            for marker in (
                "deadline expired",
                "timed out",
                "timeout",
                "unavailable",
                "temporarily unavailable",
                "try again",
            )
        )
        if transient_provider_error:
            logger.warning(
                "Google multimodal inference unavailable for %s (status=%s): %s. "
                "Falling back to text-only inference.",
                image_path,
                status_code,
                exc,
            )
        else:
            logger.exception("Google multimodal inference failed for %s", image_path)
    return None


def _parse_chart_json_response(response_text: str) -> dict[str, Any] | None:
    if not response_text:
        return None
    raw = response_text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    brace_match = re.search(r"\{.*\}", raw, flags=re.S)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except Exception:
            return None
    return None


def _apply_llm_chart_fields(figure: FigureArtifact, payload: dict[str, Any]) -> None:
    figure.chart_type = _clean_str(payload.get("chart_type")) or figure.figure_type
    figure.chart_title = _clean_str(payload.get("chart_title")) or figure.caption_text
    figure.x_axis_label = _clean_str(payload.get("x_axis_label"))
    figure.y_axis_label = _clean_str(payload.get("y_axis_label"))
    figure.x_categories = _clean_str_list(payload.get("x_categories"))
    figure.series = _clean_str_list(payload.get("series"))
    figure.approx_datapoints = _coerce_datapoints(payload.get("approx_datapoints"))
    figure.trend_summary = _clean_str(payload.get("trend_summary"))
    figure.key_chart_facts = _clean_str_list(payload.get("key_chart_facts"))
    figure.numeric_extraction_confidence = _to_float(
        payload.get("numeric_extraction_confidence")
    )


def _apply_heuristic_chart_fields(figure: FigureArtifact) -> None:
    text = normalize_whitespace(f"{figure.caption_text} {figure.nearby_text}")
    lower = text.lower()
    if figure.chart_type == "unknown":
        if "bar" in lower:
            figure.chart_type = "bar"
        elif "line" in lower or "trend" in lower:
            figure.chart_type = "line"
        else:
            figure.chart_type = "unknown"

    numbers = [float(match) for match in re.findall(r"\b\d+(?:\.\d+)?\b", text)]
    if len(numbers) >= 2 and not figure.approx_datapoints:
        figure.approx_datapoints = [
            {
                "series": "derived_series",
                "x": "start",
                "y": numbers[0],
                "unit": "",
                "approximate": True,
            },
            {
                "series": "derived_series",
                "x": "end",
                "y": numbers[-1],
                "unit": "",
                "approximate": True,
            },
        ]
        figure.key_chart_facts = [
            f"Approximate range: {numbers[0]} to {numbers[-1]}",
        ]
        figure.trend_summary = (
            "Increasing trend"
            if numbers[-1] > numbers[0]
            else "Decreasing trend" if numbers[-1] < numbers[0] else "Flat trend"
        )
        figure.numeric_extraction_confidence = 0.42

    if figure.chart_title == "":
        figure.chart_title = figure.caption_text
    if figure.chart_parse_status == "partial" and figure.approx_datapoints:
        figure.chart_parse_status = "partial"


def _chart_summary_text(figure: FigureArtifact) -> str:
    pieces = []
    if figure.chart_title:
        pieces.append(f"title={figure.chart_title}")
    if figure.x_axis_label or figure.y_axis_label:
        pieces.append(f"axes=({figure.x_axis_label}, {figure.y_axis_label})")
    if figure.series:
        pieces.append("series=" + ", ".join(figure.series[:6]))
    if figure.trend_summary:
        pieces.append(f"trend={figure.trend_summary}")
    if figure.key_chart_facts:
        pieces.extend(figure.key_chart_facts[:4])
    return " | ".join(piece for piece in pieces if piece)


def _clean_str(value: Any) -> str:
    return normalize_whitespace(str(value or ""))


def _clean_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        item_text = _clean_str(item)
        if item_text and item_text not in cleaned:
            cleaned.append(item_text)
    return cleaned


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
        return max(0.0, min(1.0, numeric))
    except Exception:
        return None


def _coerce_datapoints(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    datapoints: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        y_val: float | None = None
        try:
            y_val = float(item.get("y"))
        except Exception:
            pass
        if y_val is None:
            continue
        datapoints.append(
            {
                "series": _clean_str(item.get("series")) or "unknown_series",
                "x": _clean_str(item.get("x")),
                "y": y_val,
                "unit": _clean_str(item.get("unit")),
                "approximate": bool(item.get("approximate", True)),
            }
        )
    return datapoints


# ---------------------------------------------------------------------------
# Holistic page-level multimodal analysis
# ---------------------------------------------------------------------------


def analyze_page_screenshots(page_manifests: list[PageManifest]) -> None:
    """Run multimodal LLM analysis on full page screenshots for non-trivial pages."""
    if not settings.ENABLE_MULTIMODAL_CAPTIONING:
        return
    if not _llm_available_for_pipeline():
        return

    eligible = [
        m
        for m in page_manifests
        if m.page_class in {"visual_heavy_page", "table_heavy_page", "hard_page"}
        and m.screenshot_path
    ]
    max_pages = max(1, int(settings.LLM_CAPTION_MAX_PAGES))
    eligible = sorted(eligible, key=lambda m: m.complexity_score, reverse=True)[
        :max_pages
    ]

    def _run_screenshot_job(manifest: PageManifest) -> str | None:
        return _run_google_multimodal_inference(
            PAGE_SCREENSHOT_PROMPT, manifest.screenshot_path
        )

    for manifest, result, error in _run_parallel_jobs(
        eligible,
        _run_screenshot_job,
        max_workers=_parallel_worker_count(len(eligible)),
        timeout_seconds=float(settings.LLM_CAPTION_TIMEOUT_SECONDS),
    ):
        if error is not None:
            logger.error(
                "Page screenshot analysis failed for page %d",
                manifest.page_num,
                exc_info=error,
            )
            continue
        if result:
            manifest.llm_page_summary = result[: settings.REASONING_MAX_OUTPUT_CHARS]
            manifest.llm_enriched = True
            logger.info(
                "Page %d screenshot analysis complete (%d chars)",
                manifest.page_num,
                len(manifest.llm_page_summary),
            )


# ---------------------------------------------------------------------------
# Artifact-first reasoning enrichment
# ---------------------------------------------------------------------------


def run_reasoning_enrichment(
    page_manifests: list[PageManifest],
    tables: list[TableArtifact],
    figures: list[FigureArtifact],
) -> list[ReasoningArtifact]:
    return ReasoningEnrichmentService(page_manifests, tables, figures).run()


@dataclass(frozen=True, slots=True)
class ReasoningConfig:
    model: str
    max_pages: int
    max_artifacts_per_page: int
    max_output_chars: int
    timeout_seconds: float


class ReasoningEnrichmentService:
    def __init__(
        self,
        page_manifests: list[PageManifest],
        tables: list[TableArtifact],
        figures: list[FigureArtifact],
    ) -> None:
        self._page_manifests = page_manifests
        self._manifest_by_page = {
            manifest.page_num: manifest for manifest in page_manifests
        }
        self._tables_by_page = self._index_tables(tables)
        self._figures_by_page = self._index_figures(figures)
        self._config = ReasoningConfig(
            model=settings.REASONING_MODEL or settings.LLM_MODEL,
            max_pages=max(1, int(settings.REASONING_MAX_PAGES)),
            max_artifacts_per_page=max(
                1, int(settings.REASONING_MAX_ARTIFACTS_PER_PAGE)
            ),
            max_output_chars=max(200, int(settings.REASONING_MAX_OUTPUT_CHARS)),
            timeout_seconds=float(settings.REASONING_TIMEOUT_SECONDS),
        )

    def run(self) -> list[ReasoningArtifact]:
        if not settings.ENABLE_LLM_REASONING_ENRICHMENT:
            return []
        if not _llm_available_for_pipeline():
            return []

        eligible_pages = [manifest.page_num for manifest in self._eligible_manifests()]
        artifacts: list[ReasoningArtifact] = []
        for page_num, page_artifacts, error in _run_parallel_jobs(
            eligible_pages,
            self._reason_page,
            max_workers=_parallel_worker_count(len(eligible_pages)),
            timeout_seconds=self._config.timeout_seconds,
        ):
            if error is not None:
                logger.error(
                    "Reasoning enrichment failed for page %s",
                    page_num,
                    exc_info=error,
                )
                continue
            if page_artifacts:
                artifacts.extend(page_artifacts)

        logger.info(
            "Reasoning enrichment complete: %d artifacts across %d pages",
            len(artifacts),
            len(eligible_pages),
        )
        return artifacts

    def _eligible_manifests(self) -> list[PageManifest]:
        ordered = sorted(
            self._page_manifests,
            key=lambda manifest: manifest.complexity_score,
            reverse=True,
        )
        return [
            manifest
            for manifest in ordered
            if manifest.page_class != "simple_text_page"
            or numeric_density(manifest.full_page_text or "") > 0.15
        ][: self._config.max_pages]

    def _reason_page(self, page_num: int) -> list[ReasoningArtifact]:
        manifest = self._manifest_by_page.get(page_num)
        if manifest is None:
            return []

        page_tables = self._tables_by_page.get(page_num, [])[
            : self._config.max_artifacts_per_page
        ]
        page_figures = self._figures_by_page.get(page_num, [])[
            : self._config.max_artifacts_per_page
        ]

        artifacts: list[ReasoningArtifact] = []
        artifacts.extend(self._build_table_reasoning(page_num, page_tables))

        remaining = max(0, self._config.max_artifacts_per_page - len(artifacts))
        if remaining > 0:
            artifacts.extend(
                self._build_chart_reasoning(page_num, page_figures, remaining)
            )

        page_reasoning = self._build_page_reasoning(manifest, page_tables, page_figures)
        if page_reasoning is not None:
            artifacts.append(page_reasoning)

        return artifacts

    def _build_table_reasoning(
        self,
        page_num: int,
        tables: list[TableArtifact],
    ) -> list[ReasoningArtifact]:
        artifacts: list[ReasoningArtifact] = []
        for table in tables:
            prompt = TABLE_REASONING_PROMPT.format(
                caption=table.caption_text,
                section_path=table.section_path,
                page_num=page_num,
                units=", ".join(table.units) if table.units else "not specified",
                content=table.normalized_table_text[:1500],
            )
            artifact = self._build_reasoning_artifact(
                prompt=prompt,
                reasoning_type="table_reasoning",
                page_num=page_num,
                source_artifact_ids=[table.table_id],
                evidence_refs={
                    "table_id": table.table_id,
                    "caption": table.caption_text,
                },
                success_confidence=0.8,
                fallback_confidence=0.5,
            )
            if artifact is not None:
                artifacts.append(artifact)
        return artifacts

    def _build_chart_reasoning(
        self,
        page_num: int,
        figures: list[FigureArtifact],
        max_items: int,
    ) -> list[ReasoningArtifact]:
        artifacts: list[ReasoningArtifact] = []
        for figure in figures:
            if len(artifacts) >= max_items:
                break
            if figure.figure_type not in {"chart", "diagram", "infographic"}:
                continue

            prompt = CHART_REASONING_PROMPT.format(
                chart_type=figure.chart_type or figure.figure_type,
                chart_title=figure.chart_title or figure.caption_text,
                caption=figure.caption_text,
                section_path=figure.section_path,
                page_num=page_num,
                x_axis=figure.x_axis_label,
                y_axis=figure.y_axis_label,
                series=", ".join(figure.series) if figure.series else "unknown",
                trend_summary=figure.trend_summary,
                key_facts="; ".join(figure.key_chart_facts[:5]),
                datapoints=str(figure.approx_datapoints[:10]),
            )
            artifact = self._build_reasoning_artifact(
                prompt=prompt,
                reasoning_type="chart_reasoning",
                page_num=page_num,
                source_artifact_ids=[figure.figure_id],
                evidence_refs={
                    "figure_id": figure.figure_id,
                    "chart_type": figure.chart_type,
                    "caption": figure.caption_text,
                },
                success_confidence=0.8,
                fallback_confidence=0.5,
            )
            if artifact is not None:
                artifacts.append(artifact)
        return artifacts

    def _build_page_reasoning(
        self,
        manifest: PageManifest,
        tables: list[TableArtifact],
        figures: list[FigureArtifact],
    ) -> ReasoningArtifact | None:
        if not tables and not figures:
            return None

        summaries: list[str] = []
        for table in tables:
            summaries.append(
                f"Table [{table.table_id[:8]}]: {table.caption_text} "
                f"({len(table.json_table)} rows, units: {', '.join(table.units[:3])})"
            )
        for figure in figures:
            summaries.append(
                f"Figure [{figure.figure_id[:8]}]: {figure.figure_type} - {figure.caption_text} "
                f"(chart_type: {figure.chart_type})"
            )

        prompt = PAGE_REASONING_PROMPT.format(
            page_num=manifest.page_num,
            page_class=manifest.page_class,
            artifact_summaries="\n".join(summaries),
            llm_page_summary=manifest.llm_page_summary or "Not available",
        )
        source_ids = [table.table_id for table in tables] + [
            figure.figure_id for figure in figures
        ]
        return self._build_reasoning_artifact(
            prompt=prompt,
            reasoning_type="page_reasoning",
            page_num=manifest.page_num,
            source_artifact_ids=source_ids,
            evidence_refs={"source_artifact_ids": source_ids},
            success_confidence=0.75,
            fallback_confidence=0.4,
        )

    def _build_reasoning_artifact(
        self,
        *,
        prompt: str,
        reasoning_type: str,
        page_num: int,
        source_artifact_ids: list[str],
        evidence_refs: dict[str, Any],
        success_confidence: float,
        fallback_confidence: float,
    ) -> ReasoningArtifact | None:
        reasoning = _run_reasoning_inference(prompt, self._config.max_output_chars)
        if not reasoning:
            return None

        parsed = _parse_chart_json_response(reasoning)
        claims = parsed.get("key_insights", []) if parsed else []
        trend = parsed.get("trend_statement", "") if parsed else ""
        merged_evidence = {
            **evidence_refs,
            "evidence": (parsed.get("evidence_refs", []) if parsed else [])[:10],
        }
        return ReasoningArtifact(
            reasoning_id=str(uuid.uuid4()),
            reasoning_type=reasoning_type,
            page_nums=[page_num],
            source_artifact_ids=source_artifact_ids,
            text="\n".join(_build_reasoning_text(parsed, trend, claims))[
                : self._config.max_output_chars
            ],
            claims=claims[:5],
            evidence_refs=merged_evidence,
            confidence=success_confidence if parsed else fallback_confidence,
            model=self._config.model,
            prompt_version=REASONING_PROMPT_VERSION,
        )

    @staticmethod
    def _index_tables(tables: list[TableArtifact]) -> dict[int, list[TableArtifact]]:
        by_page: dict[int, list[TableArtifact]] = {}
        for table in tables:
            for page_num in table.page_nums:
                by_page.setdefault(page_num, []).append(table)
        return by_page

    @staticmethod
    def _index_figures(
        figures: list[FigureArtifact],
    ) -> dict[int, list[FigureArtifact]]:
        by_page: dict[int, list[FigureArtifact]] = {}
        for figure in figures:
            by_page.setdefault(figure.page_num, []).append(figure)
        return by_page


def _build_reasoning_text(
    parsed: dict[str, Any] | None,
    trend: str,
    claims: list[str],
) -> list[str]:
    """Assemble readable text from parsed reasoning JSON."""
    parts: list[str] = []
    if trend:
        parts.append(f"Trend: {trend}")
    for claim in claims[:5]:
        parts.append(f"- {claim}")
    if parsed:
        for comp in parsed.get("metric_comparisons", [])[:3]:
            parts.append(f"Comparison: {comp}")
        for caveat in parsed.get("caveats", [])[:2]:
            parts.append(f"Caveat: {caveat}")
    return parts


def _run_reasoning_inference(
    prompt: str,
    max_output: int,
) -> str | None:
    """Run LLM inference for reasoning enrichment with single retry."""
    try:
        response = LlamaSettings.llm.complete(prompt)
        text = normalize_whitespace(str(response))
        return text[:max_output] if text else None
    except Exception:
        logger.exception("Reasoning inference failed, retrying once")
        try:
            response = LlamaSettings.llm.complete(prompt)
            text = normalize_whitespace(str(response))
            return text[:max_output] if text else None
        except Exception:
            logger.exception("Reasoning inference retry also failed")
            return None
