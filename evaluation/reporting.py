"""
Helpers for turning eval summaries into API/UI-friendly payloads.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def build_quality_payload(
    retrieval_summary: Any | None = None,
    ingestion_summary: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "retrieval": _coerce(retrieval_summary),
        "ingestion": _coerce(ingestion_summary),
    }
    if extra:
        payload["extra"] = extra
    return payload


def _coerce(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value
