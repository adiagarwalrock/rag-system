"""
Health check routes.
"""

from fastapi import APIRouter

from app.services.status_service import RuntimeStatusService

router = APIRouter()


@router.get("/")
def health_check():
    return {"status": "ok", "version": "1.0.0", "service": "rag-system-backend"}


@router.get("/status")
def runtime_status():
    """Full runtime diagnostics: database, Qdrant collections, AI provider, and parsers."""
    svc = RuntimeStatusService()
    return svc.get_status()


@router.get("/models")
def list_models():
    """Return all available models from the registry.

    Response shape:
      {
        "models": [{"id": str, "default": bool}],
        "configured_default": str,
        "embedding_models": [{"id": str, "provider": str, "dimensions": int,
                               "display_name": str, "default": bool}]
      }
    """
    svc = RuntimeStatusService()
    return svc.list_available_models()
