from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

_Model = TypeVar("_Model", bound=BaseModel)


def coerce_structured_output(output: Any, output_cls: type[_Model]) -> _Model | None:
    """Best-effort coercion into a Pydantic structured-output class."""
    if isinstance(output, output_cls):
        return output

    raw = getattr(output, "raw", None)
    if isinstance(raw, output_cls):
        return raw

    if isinstance(raw, BaseModel):
        try:
            return output_cls.model_validate(raw.model_dump())
        except Exception:
            pass

    if isinstance(output, BaseModel):
        try:
            return output_cls.model_validate(output.model_dump())
        except Exception:
            pass

    if isinstance(raw, dict):
        try:
            return output_cls.model_validate(raw)
        except Exception:
            pass

    if isinstance(output, dict):
        try:
            return output_cls.model_validate(output)
        except Exception:
            pass

    message = getattr(output, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if isinstance(content, str) and content.strip():
        try:
            return output_cls.model_validate_json(content)
        except Exception:
            pass

    if isinstance(raw, str) and raw.strip():
        try:
            return output_cls.model_validate_json(raw)
        except Exception:
            pass

    if isinstance(output, str) and output.strip():
        try:
            return output_cls.model_validate_json(output)
        except Exception:
            pass

    return None
