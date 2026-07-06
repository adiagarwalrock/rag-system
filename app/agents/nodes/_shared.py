from __future__ import annotations

from typing import Any


def emit_status(state: dict[str, Any], msg: str) -> None:
    cb = state.get("status_callback")
    if cb is not None:
        try:
            cb(msg)
        except Exception:
            pass
