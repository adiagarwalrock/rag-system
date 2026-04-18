from __future__ import annotations

import re
import uuid
from typing import Any

from app.ingestion.pdf_pipeline.contracts import PageStructureStage
from app.ingestion.pdf_pipeline.helpers import (
    has_chart_signals,
    has_table_signals,
    normalize_whitespace,
    to_float_bbox,
)
from app.ingestion.pdf_pipeline.models import PageManifest, Region, ZONE_ORDER


class DefaultPageStructureStage(PageStructureStage):
    def build(
        self,
        *,
        document_id: str,
        liteparse_pages: list[dict[str, Any]],
        pymupdf_pages: list[dict[str, Any]],
        parse_meta: dict[str, Any],
        repair_meta: dict[str, Any],
    ) -> tuple[list[PageManifest], list[Region]]:
        return build_page_manifests_and_regions(
            document_id=document_id,
            liteparse_pages=liteparse_pages,
            pymupdf_pages=pymupdf_pages,
            parse_meta=parse_meta,
            repair_meta=repair_meta,
        )


def build_page_manifests_and_regions(
    document_id: str,
    liteparse_pages: list[dict[str, Any]],
    pymupdf_pages: list[dict[str, Any]],
    parse_meta: dict[str, Any],
    repair_meta: dict[str, Any],
) -> tuple[list[PageManifest], list[Region]]:
    manifests: list[PageManifest] = []
    all_regions: list[Region] = []

    liteparse_by_page = {
        int(page.get("page_num", idx + 1)): page
        for idx, page in enumerate(liteparse_pages)
    }
    current_section = "Document"

    for idx, page in enumerate(pymupdf_pages, start=1):
        page_num = int(page.get("page_num", idx))
        liteparse_page = liteparse_by_page.get(page_num, {})
        layout_predictions = _prepare_layout_predictions(
            page.get("layout_predictions") or []
        )

        page_width = float(page.get("width") or liteparse_page.get("page_width") or 0.0)
        page_height = float(
            page.get("height") or liteparse_page.get("page_height") or 0.0
        )
        raw_blocks = list(page.get("blocks") or [])
        if not raw_blocks:
            raw_blocks = _blocks_from_liteparse_items(
                liteparse_page.get("text_items") or []
            )

        page_regions: list[Region] = []
        for block in raw_blocks:
            text = normalize_whitespace(str(block.get("text") or ""))
            if not text:
                continue

            bbox = to_float_bbox(block.get("bbox"))
            layout_hint = _find_layout_hint(bbox, layout_predictions)
            region_type = _classify_region(text, bbox, page_height, layout_hint)
            if region_type in {"title", "section_heading", "subsection_heading"}:
                current_section = _sanitize_heading(text)

            page_regions.append(
                Region(
                    region_id=str(uuid.uuid4()),
                    page_num=page_num,
                    bbox=bbox,
                    region_type=region_type,
                    text=text,
                    reading_order=0,
                    section_path=current_section,
                    parent_region_id=None,
                    adjacent_region_ids=[],
                    asset_refs=[],
                    source_parser=str(block.get("source_parser", "pymupdf")),
                    confidence=_region_confidence(region_type, layout_hint),
                    zone=_infer_zone(
                        bbox,
                        page_width,
                        page_height,
                        region_type,
                        layout_hint,
                    ),
                )
            )

        ordered_regions = _order_regions(page_regions)
        for order_idx, region in enumerate(ordered_regions, start=1):
            region.reading_order = order_idx
            adjacent: list[str] = []
            if order_idx > 1:
                adjacent.append(ordered_regions[order_idx - 2].region_id)
            if order_idx < len(ordered_regions):
                adjacent.append(ordered_regions[order_idx].region_id)
            region.adjacent_region_ids = adjacent
        all_regions.extend(ordered_regions)

        complexity = _complexity_score(
            regions=ordered_regions,
            table_count=len(page.get("table_candidates") or []),
            image_count=len(page.get("image_refs") or []),
            vector_count=int(page.get("vector_count") or 0),
        )

        parser_sources = [
            str(liteparse_page.get("parser_source", "liteparse_fallback")),
            "pymupdf",
        ]
        if page.get("layout_predictions"):
            parser_sources.append("pymupdf_layout")

        manifests.append(
            PageManifest(
                document_id=document_id,
                page_num=page_num,
                page_width=page_width,
                page_height=page_height,
                screenshot_path=str(liteparse_page.get("screenshot_path", "")),
                full_page_text=str(liteparse_page.get("full_page_text", "")),
                layout_confidence=max(0.2, 1.0 - complexity * 0.35),
                complexity_score=complexity,
                page_class=_classify_page(complexity, page, ordered_regions),
                ocr_used=bool(liteparse_page.get("ocr_used", False)),
                parser_sources=parser_sources,
                layout_engine=str(parse_meta.get("layout_engine") or "pymupdf_native"),
                page_parse_degraded=bool(
                    parse_meta.get("page_parse_degraded")
                    or page.get("page_parse_degraded")
                ),
                degraded_stages=list(page.get("degraded_stages") or []),
                pdf_repair_attempted=bool(
                    repair_meta.get("pdf_repair_attempted", False)
                ),
                pdf_repair_method=repair_meta.get("pdf_repair_method"),
                pdf_repair_success=bool(repair_meta.get("pdf_repair_success", False)),
                pdf_repair_error=repair_meta.get("pdf_repair_error"),
            )
        )

    return manifests, all_regions


def _blocks_from_liteparse_items(text_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not text_items:
        return []

    rows: dict[float, list[dict[str, Any]]] = {}
    for item in text_items:
        y = float(item.get("y", 0.0))
        key = round(y / 6) * 6
        rows.setdefault(key, []).append(item)

    blocks: list[dict[str, Any]] = []
    for key in sorted(rows):
        row_items = sorted(rows[key], key=lambda item: float(item.get("x", 0.0)))
        text = " ".join(
            normalize_whitespace(str(item.get("text") or "")) for item in row_items
        ).strip()
        if not text:
            continue
        x0 = min(float(item.get("x", 0.0)) for item in row_items)
        y0 = min(float(item.get("y", 0.0)) for item in row_items)
        x1 = max(
            float(item.get("x", 0.0)) + float(item.get("width", 0.0))
            for item in row_items
        )
        y1 = max(
            float(item.get("y", 0.0)) + float(item.get("height", 0.0))
            for item in row_items
        )
        blocks.append(
            {
                "bbox": [x0, y0, x1, y1],
                "text": text,
                "source_parser": "liteparse_items",
            }
        )
    return blocks


def _classify_region(
    text: str,
    bbox: list[float],
    page_height: float,
    layout_hint: dict[str, Any] | None = None,
) -> str:
    if _looks_like_page_number(text) and bbox[1] >= page_height * 0.88:
        region_type = "footer"
    elif bbox[3] <= page_height * 0.12 and _looks_like_header_text(text):
        region_type = "header"
    elif _is_caption(text):
        region_type = "caption"
    elif _looks_like_footnote(text) or bbox[1] >= page_height * 0.9:
        region_type = "footnote"
    elif has_table_signals(text):
        region_type = "table_region"
    elif _looks_like_heading(text):
        region_type = "section_heading"
    elif has_chart_signals(text):
        region_type = "figure_region"
    else:
        region_type = "body_text"

    hinted = (layout_hint or {}).get("mapped_region_type")
    return _resolve_region_type(region_type, hinted)


def _infer_zone(
    bbox: list[float],
    page_width: float,
    page_height: float,
    region_type: str,
    layout_hint: dict[str, Any] | None = None,
) -> str:
    hinted_zone = (layout_hint or {}).get("mapped_zone")
    if hinted_zone:
        return hinted_zone

    if region_type == "header":
        return "header_zone"
    if region_type == "footer":
        return "footer_zone"
    if region_type == "footnote":
        return "footnote_zone"
    if region_type == "table_region":
        return "table_zone"
    if region_type in {"figure_region", "caption"}:
        return "figure_zone"

    x0, y0, x1, _ = bbox
    center_x = (x0 + x1) / 2.0

    if y0 >= page_height * 0.86:
        return "footnote_zone"
    if center_x < page_width * 0.45 and x1 < page_width * 0.7:
        return "left_column"
    if center_x > page_width * 0.55 and x0 > page_width * 0.3:
        return "right_column"
    if (x1 - x0) <= page_width * 0.3 and x0 > page_width * 0.65:
        return "sidebar"
    return "main_body"


def _order_regions(regions: list[Region]) -> list[Region]:
    return sorted(
        regions,
        key=lambda region: (
            ZONE_ORDER.get(region.zone, 99),
            round(region.bbox[1], 2),
            round(region.bbox[0], 2),
        ),
    )


def _complexity_score(
    regions: list[Region],
    table_count: int,
    image_count: int,
    vector_count: int,
) -> float:
    score = 0.0
    zones = {region.zone for region in regions}

    if "left_column" in zones and "right_column" in zones:
        score += 0.25
    if table_count > 0:
        score += 0.3
    if image_count > 0 or vector_count >= 20:
        score += 0.25
    if len(regions) >= 30:
        score += 0.2

    return min(1.0, score)


def _classify_page(
    complexity: float,
    page: dict[str, Any],
    regions: list[Region],
) -> str:
    table_count = len(page.get("table_candidates") or [])
    image_count = len(page.get("image_refs") or [])
    vector_count = int(page.get("vector_count") or 0)

    if table_count > 0 and complexity >= 0.4:
        return "table_heavy_page"
    if image_count > 0 or vector_count >= 20:
        return "visual_heavy_page"
    if complexity >= 0.65 or len(regions) >= 45:
        return "hard_page"
    return "simple_text_page"


def _region_confidence(
    region_type: str,
    layout_hint: dict[str, Any] | None = None,
) -> float:
    if region_type in {"body_text", "section_heading", "subsection_heading"}:
        base = 0.9
    elif region_type in {"caption", "table_region", "figure_region"}:
        base = 0.82
    elif region_type in {"footnote", "header", "footer"}:
        base = 0.72
    else:
        base = 0.75
    if layout_hint and (
        layout_hint.get("mapped_region_type") or layout_hint.get("mapped_zone")
    ):
        return min(0.98, base + 0.08)
    return base


def _prepare_layout_predictions(
    raw_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for prediction in raw_predictions:
        bbox = to_float_bbox(prediction.get("bbox"))
        if _bbox_area(bbox) <= 1.0:
            continue
        label = normalize_whitespace(str(prediction.get("label") or "")).lower()
        mapped_region_type, mapped_zone = _map_layout_label(label)
        prepared.append(
            {
                "bbox": bbox,
                "label": label,
                "mapped_region_type": mapped_region_type,
                "mapped_zone": mapped_zone,
            }
        )
    return prepared


def _find_layout_hint(
    bbox: list[float],
    predictions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not predictions:
        return None

    bbox_area = _bbox_area(bbox)
    if bbox_area <= 0:
        return None

    best: dict[str, Any] | None = None
    best_score = 0.0
    for prediction in predictions:
        pred_bbox = prediction["bbox"]
        intersection = _bbox_intersection_area(bbox, pred_bbox)
        if intersection <= 0:
            continue
        coverage = intersection / bbox_area
        score = max(_bbox_iou(bbox, pred_bbox), coverage)
        if score > best_score:
            best_score = score
            best = prediction

    if best and best_score >= 0.12:
        return best
    return None


def _map_layout_label(label: str) -> tuple[str | None, str | None]:
    if not label:
        return None, None

    if "table" in label:
        return "table_region", "table_zone"
    if any(token in label for token in ("caption",)):
        return "caption", "figure_zone"
    if any(
        token in label
        for token in ("figure", "chart", "graph", "plot", "image", "diagram")
    ):
        return "figure_region", "figure_zone"
    if "footnote" in label or label.startswith("note"):
        return "footnote", "footnote_zone"
    if "footer" in label:
        return "footer", "footer_zone"
    if "header" in label:
        return "header", "header_zone"
    if "title" in label:
        return "title", "header_zone"
    if any(token in label for token in ("heading", "section", "subtitle")):
        return "section_heading", "main_body"
    if any(token in label for token in ("left_column", "left column", "column_1", "col1")):
        return "body_text", "left_column"
    if any(
        token in label for token in ("right_column", "right column", "column_2", "col2")
    ):
        return "body_text", "right_column"
    if "sidebar" in label:
        return "body_text", "sidebar"
    if any(token in label for token in ("body", "text", "paragraph")):
        return "body_text", "main_body"
    return None, None


def _resolve_region_type(current: str, hinted: str | None) -> str:
    if not hinted:
        return current

    high_priority = {
        "table_region",
        "figure_region",
        "caption",
        "footnote",
        "header",
        "footer",
        "title",
    }
    if hinted in high_priority:
        return hinted

    if hinted in {"section_heading", "subsection_heading"} and current in {
        "body_text",
        "section_heading",
        "subsection_heading",
    }:
        return hinted

    if hinted == "body_text" and current in {
        "body_text",
        "section_heading",
        "subsection_heading",
    }:
        return hinted

    return current


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_intersection_area(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _bbox_iou(left: list[float], right: list[float]) -> float:
    intersection = _bbox_intersection_area(left, right)
    if intersection <= 0:
        return 0.0
    union = _bbox_area(left) + _bbox_area(right) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _looks_like_page_number(text: str) -> bool:
    return bool(
        re.match(r"^\s*(page\s+)?\d+(\s*/\s*\d+)?\s*$", text.strip(), re.IGNORECASE)
    )


def _looks_like_footnote(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(re.match(r"^(\*|\u2020|\u2021|\[\d+\]|\d+[.)])\s+", stripped))


def _looks_like_header_text(text: str) -> bool:
    if len(text) > 80:
        return False
    return bool(re.search(r"[A-Za-z]", text))


def _is_caption(text: str) -> bool:
    stripped = text.strip().lower()
    return bool(re.match(r"^(figure|fig\.|table|chart|graph|exhibit)\s*\d*", stripped))


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > 120 or stripped.endswith((".", ";", ":")):
        return False
    words = [word for word in re.split(r"\s+", stripped) if word]
    if len(words) > 14:
        return False
    if has_table_signals(stripped):
        return False

    titled = sum(1 for word in words if word[0].isupper())
    return (titled / len(words)) >= 0.55


def _sanitize_heading(text: str) -> str:
    value = normalize_whitespace(text)
    value = re.sub(r"\s*[:\-]+\s*$", "", value)
    return value or "Document"
