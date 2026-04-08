"""
Health check route.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def health_check():
    return {"status": "ok", "version": "1.0.0", "service": "rag-system-backend"}
