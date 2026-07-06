from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.config import settings

router = APIRouter()

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


@router.get("/image")
def get_artifact_image(path: str = Query(..., min_length=1)):
    root = Path(settings.PARSED_ARTIFACTS_DIR).expanduser().resolve()
    requested = Path(path).expanduser()
    candidate = requested if requested.is_absolute() else root / requested

    try:
        resolved = candidate.resolve()
    except OSError:
        raise HTTPException(status_code=400, detail="Invalid artifact path")

    if root != resolved and root not in resolved.parents:
        raise HTTPException(status_code=400, detail="Artifact path is outside parsed artifacts")
    if resolved.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Artifact is not a supported image")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Artifact image not found")

    return FileResponse(resolved)
