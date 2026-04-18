from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymupdf as fitz
from liteparse import LiteParse

from app.core.config import settings
from app.ingestion.pdf_pipeline.contracts import ExtractionResult, PDFExtractionStage
from app.ingestion.pdf_pipeline.helpers import (
    normalize_block_text,
    normalize_whitespace,
    to_float_bbox,
)
from app.ingestion.pdf_pipeline.registry import PDFPipelineRegistry

logger = logging.getLogger(__name__)

_TABLE_DETECTION_STRATEGIES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("default", {}),
    ("lines", {"strategy": "lines"}),
    ("text", {"strategy": "text"}),
)


class DefaultPDFExtractionStage(PDFExtractionStage):
    def extract(
        self,
        *,
        pdf_doc: fitz.Document,
        file_path: str,
        parse_input_path: str,
        screenshot_dir: Path,
    ) -> ExtractionResult:
        liteparse_pages, liteparse_meta = extract_liteparse_pages(
            parse_input_path,
            screenshot_dir=screenshot_dir,
        )
        if not liteparse_pages:
            liteparse_pages = build_liteparse_fallback_pages(pdf_doc, screenshot_dir)

        pymupdf_pages, pymupdf_meta = extract_pymupdf_pages(pdf_doc, parse_input_path)
        parse_meta = {
            "layout_engine": (
                liteparse_meta.get("layout_engine")
                if liteparse_pages
                and liteparse_meta.get("layout_engine") == "liteparse"
                else pymupdf_meta.get("layout_engine", "pymupdf_native")
            ),
            "layout_engine_fallback_reason": liteparse_meta.get(
                "layout_engine_fallback_reason"
            ),
            "page_parse_degraded": bool(
                liteparse_meta.get("page_parse_degraded")
                or pymupdf_meta.get("page_parse_degraded")
            ),
        }
        return ExtractionResult(
            liteparse_pages=liteparse_pages,
            pymupdf_pages=pymupdf_pages,
            parse_meta=parse_meta,
        )


def extract_liteparse_pages(
    file_path: str,
    screenshot_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Primary layout/text extraction using LiteParse Python API."""
    pages: list[dict[str, Any]] = []
    meta = {
        "layout_engine": "liteparse",
        "page_parse_degraded": False,
        "layout_engine_fallback_reason": None,
    }

    parser = LiteParse(install_if_not_available=False)
    try:
        result = parser.parse(
            file_path,
            ocr_enabled=settings.ENABLE_OCR_FALLBACK,
            dpi=150,
            precise_bounding_box=True,
        )
    except Exception as exc:
        meta["layout_engine"] = "pymupdf_native"
        meta["page_parse_degraded"] = True
        meta["layout_engine_fallback_reason"] = f"liteparse_parse_failed:{exc}"
        logger.warning("LiteParse parse failed for %s: %s", file_path, exc)
        return pages, meta

    screenshot_map: dict[int, str] = {}
    try:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        screenshots = parser.screenshot(file_path, output_dir=screenshot_dir, dpi=120)
        for shot in screenshots.screenshots:
            screenshot_map[int(shot.page_num)] = shot.image_path
    except Exception as exc:
        meta["page_parse_degraded"] = True
        logger.warning("LiteParse screenshot failed for %s: %s", file_path, exc)

    for page in result.pages:
        page_ocr_used = _resolve_page_ocr_used(page)
        text_items = []
        for item in page.textItems or []:
            text = str(item.text or "").strip()
            if not text:
                continue
            text_items.append(
                {
                    "x": float(item.x),
                    "y": float(item.y),
                    "width": float(item.width),
                    "height": float(item.height),
                    "text": text,
                }
            )

        page_num = int(page.pageNum)
        pages.append(
            {
                "page_num": page_num,
                "full_page_text": str(page.text or ""),
                "text_items": text_items,
                "page_width": float(page.width),
                "page_height": float(page.height),
                "screenshot_path": screenshot_map.get(page_num, ""),
                "ocr_used": page_ocr_used,
                "parser_source": "liteparse",
            }
        )

    return pages, meta


def build_liteparse_fallback_pages(
    pdf_doc: fitz.Document,
    screenshot_dir: Path,
) -> list[dict[str, Any]]:
    """Fallback page payload when LiteParse is unavailable."""
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []

    for page_idx, page in enumerate(pdf_doc, start=1):
        screenshot_path = screenshot_dir / f"page_{page_idx}.png"
        screenshot = ""
        try:
            pix = page.get_pixmap(dpi=120)
            pix.save(str(screenshot_path))
            screenshot = str(screenshot_path)
        except Exception:
            logger.warning(
                "Fallback screenshot generation failed for page %s", page_idx
            )

        words = page.get_text("words") or []
        text_items = []
        for word in words:
            if len(word) < 5:
                continue
            x0, y0, x1, y1, text = word[:5]
            token = str(text).strip()
            if not token:
                continue
            text_items.append(
                {
                    "x": float(x0),
                    "y": float(y0),
                    "width": float(x1 - x0),
                    "height": float(y1 - y0),
                    "text": token,
                }
            )

        pages.append(
            {
                "page_num": page_idx,
                "full_page_text": page.get_text("text") or "",
                "text_items": text_items,
                "page_width": float(page.rect.width),
                "page_height": float(page.rect.height),
                "screenshot_path": screenshot,
                "ocr_used": False,
                "parser_source": "liteparse_fallback",
            }
        )

    return pages


def extract_pymupdf_pages(
    pdf_doc: fitz.Document,
    _file_path: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Object-level extraction for tables/images/vectors, with page-level degradation flags."""
    pages: list[dict[str, Any]] = []
    meta = {"layout_engine": "pymupdf_native", "page_parse_degraded": False}
    layout_enabled = bool(getattr(settings, "ENABLE_PYMUPDF_LAYOUT", True))
    if layout_enabled:
        layout_enabled = (
            PDFPipelineRegistry().layout_engine_manager.ensure_layout_activation()
        )
    layout_used = False

    for page_idx, page in enumerate(pdf_doc, start=1):
        components, degraded_stages = _extract_page_components(page)
        page_degraded = bool(degraded_stages)

        layout_predictions = _extract_layout_predictions(page) if layout_enabled else []
        if layout_predictions:
            layout_used = True

        pages.append(
            {
                "page_num": page_idx,
                "width": float(page.rect.width),
                "height": float(page.rect.height),
                "rotation": int(page.rotation),
                "blocks": components["blocks"],
                "words": components["words"],
                "image_refs": components["image_refs"],
                "vector_bboxes": components["vector_bboxes"],
                "vector_count": len(components["vector_bboxes"]),
                "table_candidates": components["table_candidates"],
                "layout_predictions": layout_predictions,
                "page_parse_degraded": page_degraded,
                "degraded_stages": degraded_stages,
            }
        )

        if page_degraded:
            meta["page_parse_degraded"] = True

    if layout_used:
        meta["layout_engine"] = "pymupdf_layout"

    return pages, meta


def _extract_page_components(page: fitz.Page) -> tuple[dict[str, list[Any]], list[str]]:
    components: dict[str, list[Any]] = {}
    degraded_stages: list[str] = []
    extractors: tuple[tuple[str, Callable[[fitz.Page], list[Any] | None], str], ...] = (
        ("blocks", _extract_blocks, "text_blocks"),
        ("words", _extract_words, "words"),
        ("image_refs", _extract_page_images, "images"),
        ("vector_bboxes", _extract_vector_bboxes, "vectors"),
        ("table_candidates", _extract_table_candidates, "table_candidates"),
    )

    for key, extractor, stage_name in extractors:
        value = extractor(page)
        if value is None:
            components[key] = []
            degraded_stages.append(stage_name)
            continue
        components[key] = value

    return components, degraded_stages


def _extract_blocks(page: fitz.Page) -> list[dict[str, Any]] | None:
    try:
        blocks: list[dict[str, Any]] = []
        for block in page.get_text("blocks") or []:
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            normalized = normalize_block_text(str(text))
            if not normalized:
                continue
            blocks.append(
                {
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "text": normalized,
                }
            )
        return blocks
    except Exception:
        logger.exception("PyMuPDF block extraction failed")
        return None


def _extract_words(page: fitz.Page) -> list[dict[str, Any]] | None:
    try:
        words: list[dict[str, Any]] = []
        for word in page.get_text("words") or []:
            if len(word) < 5:
                continue
            x0, y0, x1, y1, text = word[:5]
            token = normalize_whitespace(str(text))
            if not token:
                continue
            words.append(
                {
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "text": token,
                }
            )
        return words
    except Exception:
        logger.exception("PyMuPDF word extraction failed")
        return None


def _extract_page_images(page: fitz.Page) -> list[dict[str, Any]] | None:
    try:
        images: list[dict[str, Any]] = []
        for index, image in enumerate(page.get_images(full=True), start=1):
            xref = image[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            for rect in rects:
                images.append(
                    {
                        "image_ref": f"img_{index}",
                        "xref": int(xref),
                        "bbox": [
                            float(rect.x0),
                            float(rect.y0),
                            float(rect.x1),
                            float(rect.y1),
                        ],
                    }
                )
        return images
    except Exception:
        logger.exception("PyMuPDF image extraction failed")
        return None


def _extract_vector_bboxes(page: fitz.Page) -> list[list[float]] | None:
    try:
        vectors: list[list[float]] = []
        for drawing in page.get_drawings() or []:
            rect = drawing.get("rect")
            if rect is None:
                continue
            vectors.append(
                [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
            )
        return vectors
    except Exception:
        logger.exception("PyMuPDF vector extraction failed")
        return None


def _extract_table_candidates(page: fitz.Page) -> list[dict[str, Any]] | None:
    page_num = _page_number_for_logs(page)
    successful_detection = False
    strategy_failures = 0
    last_failure: tuple[str, Exception] | None = None

    for strategy_name, kwargs in _TABLE_DETECTION_STRATEGIES:
        try:
            finder = page.find_tables(**kwargs)
            successful_detection = True
        except Exception as exc:
            strategy_failures += 1
            last_failure = (strategy_name, exc)
            logger.debug(
                "PyMuPDF table detection strategy failed (page=%d strategy=%s error=%s: %s)",
                page_num,
                strategy_name,
                type(exc).__name__,
                exc,
            )
            continue

        try:
            candidates = _build_table_candidates_from_tables(
                tables=getattr(finder, "tables", []) if finder is not None else [],
                strategy_name=strategy_name,
                page_num=page_num,
            )
        except Exception as exc:
            strategy_failures += 1
            last_failure = (strategy_name, exc)
            logger.debug(
                "PyMuPDF table candidate normalization failed (page=%d strategy=%s error=%s: %s)",
                page_num,
                strategy_name,
                type(exc).__name__,
                exc,
            )
            continue
        if candidates:
            return candidates

    if not successful_detection and strategy_failures:
        strategy_label = last_failure[0] if last_failure is not None else "unknown"
        failure_label = (
            type(last_failure[1]).__name__ if last_failure is not None else "unknown"
        )
        failure_message = str(last_failure[1]) if last_failure is not None else "n/a"
        logger.warning(
            "PyMuPDF table detection failed for page %d after %d strategies (last_strategy=%s error=%s: %s)",
            page_num,
            strategy_failures,
            strategy_label,
            failure_label,
            failure_message,
        )
        return None

    return []


def _build_table_candidates_from_tables(
    *,
    tables: list[Any],
    strategy_name: str,
    page_num: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, table in enumerate(tables, start=1):
        try:
            bbox = to_float_bbox(getattr(table, "bbox", None))
        except Exception as exc:
            logger.warning(
                "Skipping malformed PyMuPDF table candidate (page=%d strategy=%s candidate_index=%d stage=bbox error=%s: %s)",
                page_num,
                strategy_name,
                index,
                type(exc).__name__,
                exc,
            )
            continue
        try:
            rows = table.extract() or []
        except Exception:
            rows = []

        normalized_rows: list[list[str]] = []
        for row in rows:
            if not row:
                continue
            normalized_rows.append(
                [normalize_whitespace(str(cell or "")) for cell in row]
            )
        if not normalized_rows:
            continue

        candidates.append(
            {
                "candidate_id": f"{strategy_name}_table_candidate_{index}",
                "bbox": bbox,
                "rows": normalized_rows,
            }
        )
    return candidates


def _page_number_for_logs(page: fitz.Page) -> int:
    page_index = getattr(page, "number", None)
    if isinstance(page_index, int):
        return page_index + 1
    return -1


def _extract_layout_predictions(page: fitz.Page) -> list[dict[str, Any]]:
    getter = getattr(fitz, "_get_layout", None)
    if not callable(getter):
        return []

    try:
        layout_items = getter(page)
    except Exception:
        logger.debug("pymupdf.layout predictions unavailable for page")
        return []

    predictions: list[dict[str, Any]] = []
    for item in layout_items or []:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            continue
        x0, y0, x1, y1, label = item[:5]
        predictions.append(
            {
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "label": normalize_whitespace(str(label or "")),
            }
        )
    return predictions


def _resolve_page_ocr_used(page: Any) -> bool:
    """Best-effort extraction of per-page OCR usage from LiteParse page objects."""
    candidate_fields = (
        "ocr_used",
        "ocrUsed",
        "used_ocr",
        "usedOCR",
        "is_ocr",
        "isOCR",
        "ocr",
    )
    for field_name in candidate_fields:
        value = getattr(page, field_name, None)
        parsed = _coerce_bool(value)
        if parsed is not None:
            return parsed

    metadata = getattr(page, "metadata", None)
    if isinstance(metadata, dict):
        for field_name in candidate_fields:
            parsed = _coerce_bool(metadata.get(field_name))
            if parsed is not None:
                return parsed

    return False


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = normalize_whitespace(value).lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None
