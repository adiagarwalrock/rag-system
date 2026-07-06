"""
Backfill page screenshots for Reducto-parsed documents already in Qdrant.

For every point where parser_name == "reducto" and asset_refs is empty:
  1. Looks up the raw PDF path from the SQL database via Document.storage_path
  2. Renders page screenshots with PyMuPDF into
     artifacts/parsed/{document_id}/screenshots/page_{N}.png
  3. Patches the Qdrant payload (top-level asset_refs + inside _node_content JSON)
     using set_payload — no vectors re-uploaded.

Run:
    uv run python -m app.scripts.backfill_reducto_screenshots
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import pymupdf as fitz
import qdrant_client
from qdrant_client.http import models as qm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import settings
from app.core.logging_config import configure_logging
from app.db.models.document import Document
from app.db.snowflake import SessionLocal

warnings.filterwarnings(
    "ignore",
    message="Api key is used with an insecure connection.",
    category=UserWarning,
)

configure_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_SCREENSHOT_DPI: int = 120  # matches adapters.py page screenshot DPI
_BATCH_SIZE: int = 500


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------


def _make_client() -> qdrant_client.QdrantClient:
    return qdrant_client.QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )


def _scroll_reducto_points(
    client: qdrant_client.QdrantClient, collection: str
) -> list:
    """Return all Qdrant points with parser_name=reducto and empty asset_refs."""
    scroll_filter = qm.Filter(
        must=[
            qm.FieldCondition(key="parser_name", match=qm.MatchValue(value="reducto")),
        ]
    )
    points: list = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=_BATCH_SIZE,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        points.extend(batch)
        if offset is None:
            break
    # Filter empty asset_refs in Python (not indexed, no IsEmptyCondition needed)
    return [pt for pt in points if not (pt.payload or {}).get("asset_refs")]


# ---------------------------------------------------------------------------
# Screenshot rendering
# ---------------------------------------------------------------------------


def _render_screenshots(
    pdf_path: Path, document_id: str, page_nums: set[int]
) -> dict[int, str]:
    """Render PNGs for the requested page numbers; return {page_num: abs_path}."""
    screenshot_dir = Path(settings.PARSED_ARTIFACTS_DIR) / document_id / "screenshots"
    try:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Cannot create screenshot dir for %s: %s", document_id, exc)
        return {}

    result: dict[int, str] = {}
    try:
        pdf_doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.warning("fitz.open failed for %s: %s", pdf_path, exc)
        return {}

    try:
        for page_idx in range(len(pdf_doc)):
            page_num = page_idx + 1  # fitz 0-indexed; page_num is 1-indexed
            if page_num not in page_nums:
                continue
            out = screenshot_dir / f"page_{page_num}.png"
            if out.exists():
                result[page_num] = str(out)
                continue
            try:
                pdf_doc[page_idx].get_pixmap(dpi=_SCREENSHOT_DPI).save(str(out))
                result[page_num] = str(out)
            except Exception as exc:
                logger.warning("Screenshot failed for page %d of %s: %s", page_num, document_id, exc)
    finally:
        pdf_doc.close()

    return result


# ---------------------------------------------------------------------------
# Payload patching
# ---------------------------------------------------------------------------


def _patch_node_content(raw: str, new_asset_refs: list[str]) -> str:
    """Update asset_refs inside the _node_content JSON blob."""
    try:
        node = json.loads(raw)
        metadata = node.get("metadata", {})
        metadata["asset_refs"] = new_asset_refs
        node["metadata"] = metadata
        return json.dumps(node)
    except Exception:
        return raw


def _update_point(
    client: qdrant_client.QdrantClient,
    collection: str,
    point_id: str,
    asset_refs: list[str],
    raw_node_content: str | None,
) -> None:
    payload: dict = {"asset_refs": asset_refs}
    if raw_node_content:
        payload["_node_content"] = _patch_node_content(raw_node_content, asset_refs)
    client.set_payload(
        collection_name=collection,
        payload=payload,
        points=[point_id],
        wait=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    collection = settings.COLLECTION_NAME
    client = _make_client()

    if not client.collection_exists(collection):
        logger.error("Collection %s does not exist — aborting.", collection)
        return

    logger.info("Scrolling Qdrant for Reducto points with empty asset_refs ...")
    points = _scroll_reducto_points(client, collection)
    logger.info("Found %d points to backfill.", len(points))
    if not points:
        return

    # Group by document_id so we open each PDF once
    by_doc: dict[str, list] = {}
    for pt in points:
        doc_id = (pt.payload or {}).get("document_id")
        if doc_id:
            by_doc.setdefault(doc_id, []).append(pt)
        else:
            logger.warning("Point %s has no document_id — skipping.", pt.id)

    updated = 0
    skipped = 0

    with SessionLocal() as db:
        for document_id, doc_points in by_doc.items():
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc or not doc.storage_path:
                logger.warning("No DB record or storage_path for %s — skipping.", document_id)
                skipped += len(doc_points)
                continue

            pdf_path = Path(doc.storage_path)
            if not pdf_path.exists():
                logger.warning("PDF not on disk at %s — skipping %s.", pdf_path, document_id)
                skipped += len(doc_points)
                continue

            page_nums: set[int] = set()
            for pt in doc_points:
                pn = (pt.payload or {}).get("page_num")
                if pn is not None:
                    page_nums.add(int(pn))

            screenshot_map = _render_screenshots(pdf_path, document_id, page_nums)
            logger.info(
                "Document %s (%s): rendered %d/%d pages.",
                document_id, doc.name, len(screenshot_map), len(page_nums),
            )

            for pt in doc_points:
                pn = (pt.payload or {}).get("page_num")
                if pn is None or int(pn) not in screenshot_map:
                    skipped += 1
                    continue
                asset_refs = [screenshot_map[int(pn)]]
                raw_nc = (pt.payload or {}).get("_node_content")
                _update_point(client, collection, pt.id, asset_refs, raw_nc)
                updated += 1

    logger.info("Done. Updated: %d  Skipped: %d", updated, skipped)


if __name__ == "__main__":
    main()
