"""
Primitive type-coercion helpers shared across retrieval and ingestion layers.

These are intentionally simple — no domain logic, just safe conversions with a
fallback default. Keep them here and import rather than re-implementing locally.
"""
from __future__ import annotations

from typing import Any


def safe_int(value: Any, default: int | None = None) -> int | None:
    """
    Coerce *value* to ``int``.

    - No default (or ``default=None``): returns ``None`` on failure — used by
      the ingestion layer when ``None`` signals "not present".
    - Integer default: returns that value on failure — used by the retrieval
      layer when a numeric fallback is required.
    """
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Coerce *value* to ``float``; return *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool) -> bool:
    """
    Coerce *value* to ``bool``.

    Recognises common string representations (case-insensitive, leading/trailing
    whitespace stripped):
    - Truthy:  ``"true"``, ``"1"``, ``"yes"``
    - Falsy:   ``"false"``, ``"0"``, ``"no"``

    Returns *default* for any unrecognised input.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


def normalize_metric_subject(token: str) -> str:
    """
    Normalise a query token for metric-subject matching.

    Lowercases, strips leading/trailing hyphens, and naively de-pluralises
    (``-ies`` → ``-y``, trailing ``-s``).
    """
    normalized = token.lower().strip("-")
    if normalized.endswith("ies") and len(normalized) > 4:
        return f"{normalized[:-3]}y"
    if normalized.endswith("s") and len(normalized) > 3:
        return normalized[:-1]
    return normalized
