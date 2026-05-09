from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymupdf as fitz
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.core.config import settings
from app.ingestion.pdf_pipeline.contracts import ExtractionResult, PDFExtractionStage
from app.ingestion.pdf_pipeline.helpers import (
    normalize_block_text,
    normalize_whitespace,
    to_float_bbox,
)
from app.ingestion.pdf_pipeline.registry import PDFPipelineRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Docling singleton — loaded lazily on first use, reused across all documents
# ---------------------------------------------------------------------------

_DOCLING_CONVERTER: Any | None = None


def _get_docling_converter() -> DocumentConverter:
    """Return a lazily initialised Docling DocumentConverter singleton."""
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = True
        if isinstance(opts.table_structure_options, TableStructureOptions):
            opts.table_structure_options.do_cell_matching = True

        _DOCLING_CONVERTER = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        logger.info("Docling DocumentConverter initialised (DocLayNet layout model)")
    return _DOCLING_CONVERTER


def _docling_table_to_rows(table_item: Any) -> list[list[str]]:
    """Convert a Docling TableItem data grid to the list[list[str]] candidate format."""
    try:
        grid = getattr(getattr(table_item, "data", None), "grid", None)
        if not grid:
            return []
        return [
            [normalize_whitespace(getattr(cell, "text", "") or "") for cell in row]
            for row in grid
            if row
        ]
    except Exception:
        return []


def _bbox_bottomleft_to_topleft(
    bbox: Any,
    page_height: float,
) -> list[float]:
    """
    Convert a Docling BOTTOMLEFT BoundingBox to PyMuPDF TOPLEFT [x0, y0, x1, y1].

    Docling ProvenanceItem bboxes use CoordOrigin.BOTTOMLEFT (PDF spec).
    PyMuPDF uses CoordOrigin.TOPLEFT (y increases downward from top of page).
    """
    tl = bbox.to_top_left_origin(page_height=page_height)
    return [float(tl.l), float(tl.t), float(tl.r), float(tl.b)]


def extract_docling_pages(
    file_path: str,
    pdf_doc: fitz.Document,
    screenshot_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, dict[str, Any]]]:
    """
    Docling-based document understanding for table and figure extraction.

    Text content and screenshots are sourced from the PyMuPDF-based text extraction
    (build_pymupdf_text_pages) since Docling's strength is structural layout,
    not raw text extraction. The returned text_pages carry parser_source="docling"
    to signal the active engine downstream.

    Returns:
        text_pages: per-page dicts in the 8-field contract expected by page_structure.py
        meta: layout engine metadata dict
        docling_by_page: {page_num: {table_candidates, figure_candidates}} — injected into
                         pymupdf_pages by DefaultPDFExtractionStage before the artifact stage
    """
    # Build the page payload using PyMuPDF text extraction (screenshots + word items)
    text_pages = build_pymupdf_text_pages(pdf_doc, screenshot_dir)
    for page in text_pages:
        page["parser_source"] = "docling"

    meta: dict[str, Any] = {
        "layout_engine": "docling",
        "page_parse_degraded": False,
        "layout_engine_fallback_reason": None,
    }
    docling_by_page: dict[int, dict[str, Any]] = {}

    # Pre-compute page heights (points) for BOTTOMLEFT → TOPLEFT conversion
    page_heights: dict[int, float] = {
        i + 1: float(pdf_doc[i].rect.height) for i in range(len(pdf_doc))
    }

    try:
        converter = _get_docling_converter()
        result = converter.convert(file_path)
        doc = result.document

        # --- Table candidates ---
        for table in getattr(doc, "tables", []) or []:
            prov_list = getattr(table, "prov", None)
            if not prov_list:
                continue
            prov = prov_list[0]
            page_no = int(getattr(prov, "page_no", 0))
            if page_no < 1:
                continue
            raw_bbox = getattr(prov, "bbox", None)
            if raw_bbox is None:
                continue
            rows = _docling_table_to_rows(table)
            if len(rows) < 2:
                continue
            page_h = page_heights.get(page_no, 792.0)
            entry = docling_by_page.setdefault(
                page_no, {"table_candidates": [], "figure_candidates": []}
            )
            entry["table_candidates"].append(
                {
                    "candidate_id": (
                        f"docling_table_{page_no}_{len(entry['table_candidates'])}"
                    ),
                    "bbox": _bbox_bottomleft_to_topleft(raw_bbox, page_h),
                    "rows": rows,
                    "source": "docling",
                }
            )

        # --- Figure / picture candidates ---
        for picture in getattr(doc, "pictures", []) or []:
            prov_list = getattr(picture, "prov", None)
            if not prov_list:
                continue
            prov = prov_list[0]
            page_no = int(getattr(prov, "page_no", 0))
            if page_no < 1:
                continue
            raw_bbox = getattr(prov, "bbox", None)
            if raw_bbox is None:
                continue
            page_h = page_heights.get(page_no, 792.0)
            entry = docling_by_page.setdefault(
                page_no, {"table_candidates": [], "figure_candidates": []}
            )
            entry["figure_candidates"].append(
                {
                    "bbox": _bbox_bottomleft_to_topleft(raw_bbox, page_h),
                    "kind": "docling_picture",
                }
            )

    except Exception as exc:
        logger.warning(
            "Docling extraction failed for %s (%s: %s) — using PyMuPDF only",
            Path(file_path).name,
            type(exc).__name__,
            exc,
        )
        meta["page_parse_degraded"] = True
        meta["layout_engine_fallback_reason"] = f"docling_failed:{type(exc).__name__}"
        return text_pages, meta, {}

    n_tables = sum(len(v["table_candidates"]) for v in docling_by_page.values())
    n_figures = sum(len(v["figure_candidates"]) for v in docling_by_page.values())
    logger.info(
        "Docling: %d table candidates, %d figure candidates from %s",
        n_tables,
        n_figures,
        Path(file_path).name,
    )
    return text_pages, meta, docling_by_page


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
        docling_by_page: dict[int, dict[str, Any]] = {}

        text_pages, text_meta, docling_by_page = extract_docling_pages(
            parse_input_path, pdf_doc, screenshot_dir
        )

        pymupdf_pages, pymupdf_meta = extract_pymupdf_pages(pdf_doc, parse_input_path)

        # Merge Docling structural candidates into the pymupdf_pages payload.
        # Table candidates replace PyMuPDF's when Docling found any (higher structural fidelity).
        # Figure candidates are stored under a new key consumed by build_figure_artifacts().
        if docling_by_page:
            for page in pymupdf_pages:
                page_num = int(page.get("page_num", 1))
                docling_data = docling_by_page.get(page_num)
                if not docling_data:
                    continue
                if docling_data["table_candidates"]:
                    page["table_candidates"] = docling_data["table_candidates"]
                if docling_data["figure_candidates"]:
                    page["docling_figure_candidates"] = docling_data[
                        "figure_candidates"
                    ]

        parse_meta = {
            "layout_engine": text_meta.get("layout_engine", "docling"),
            "layout_engine_fallback_reason": text_meta.get(
                "layout_engine_fallback_reason"
            ),
            "page_parse_degraded": bool(
                text_meta.get("page_parse_degraded")
                or pymupdf_meta.get("page_parse_degraded")
            ),
        }
        return ExtractionResult(
            liteparse_pages=text_pages,
            pymupdf_pages=pymupdf_pages,
            parse_meta=parse_meta,
        )


def build_pymupdf_text_pages(
    pdf_doc: fitz.Document,
    screenshot_dir: Path,
) -> list[dict[str, Any]]:
    """Extract page text and screenshots using PyMuPDF."""
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []

    for page_idx in range(1, len(pdf_doc) + 1):
        page: fitz.Page = pdf_doc[page_idx - 1]
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
                "parser_source": "pymupdf",
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

    for page_idx in range(1, len(pdf_doc) + 1):
        page: fitz.Page = pdf_doc[page_idx - 1]
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
