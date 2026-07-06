"""
Shared page-screenshot helpers for external parsers.

render_page_screenshots() renders one PNG per content page from the raw PDF
using PyMuPDF at 120 DPI and saves to the parsed-artifacts directory.
screenshot_refs() maps a chunk's page numbers to the screenshot paths.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

from app.core.config import settings

_SCREENSHOT_DPI: int = 120  # matches adapters.py page screenshot DPI


def render_page_screenshots(
    pdf_path: Path,
    document_id: str | None,
    page_sections: list[tuple[int, str]],
) -> dict[int, str]:
    """Render one PNG per content page from the raw PDF; return {page_num: abs_path}.

    Returns {} when document_id is None or when fitz fails to open the PDF.
    """
    if not document_id:
        return {}
    screenshot_dir = Path(settings.PARSED_ARTIFACTS_DIR) / document_id / "screenshots"
    page_nums = {page_num for page_num, _ in page_sections}
    try:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"  [screenshots] WARNING: cannot create screenshot dir: {exc}")
        return {}
    result: dict[int, str] = {}
    try:
        pdf_doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"  [screenshots] WARNING: fitz.open failed — skipping screenshots: {exc}")
        return {}
    try:
        for page_idx in range(len(pdf_doc)):
            page_num = page_idx + 1  # fitz is 0-indexed; parsers use 1-indexed pages
            if page_num not in page_nums:
                continue
            screenshot_path = screenshot_dir / f"page_{page_num}.png"
            try:
                pix = pdf_doc[page_idx].get_pixmap(dpi=_SCREENSHOT_DPI)
                pix.save(str(screenshot_path))
                result[page_num] = str(screenshot_path)
            except Exception as exc:
                print(f"  [screenshots] WARNING: screenshot failed for page {page_num}: {exc}")
    finally:
        pdf_doc.close()
    print(f"  [screenshots] {len(result)} page screenshots rendered")
    return result


def screenshot_refs(
    page_nums: list[int], screenshot_map: dict[int, str] | None
) -> list[str]:
    """Return the screenshot paths for the given page numbers."""
    if not screenshot_map:
        return []
    return [screenshot_map[p] for p in page_nums if p in screenshot_map]
