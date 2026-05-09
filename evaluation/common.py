"""
Shared helpers for eval scoring and payload shaping.
"""

from dataclasses import asdict, is_dataclass
from statistics import mean
from typing import Any


def safe_lower(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def unique_nonempty(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = safe_lower(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(str(value))
    return result


def as_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: as_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_plain_dict(v) for v in value]
    if isinstance(value, tuple):
        return [as_plain_dict(v) for v in value]
    return value


def mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))
