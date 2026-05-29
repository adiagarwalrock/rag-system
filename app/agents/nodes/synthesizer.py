"""
synthesizer_node: thin wrapper around GroundedAnswerSynthesizer.
"""

from __future__ import annotations

import logging
from typing import Any

from app.retrieval.retriever import GroundedAnswerSynthesizer

logger = logging.getLogger(__name__)


def synthesizer_node(state: dict[str, Any]) -> dict[str, Any]:
    _emit(state, "Synthesizing grounded answer…")

    synthesizer = GroundedAnswerSynthesizer(
        client_id=state["client_id"],
        reasoning_effort=state.get("reasoning_effort", "medium"),
        reasoning_summary=state.get("reasoning_summary"),
        reasoning_callback=state.get("reasoning_callback"),
        conversation_context=state.get("conversation_context") or {},
    )

    result = synthesizer.synthesize(
        question=state["question"],
        citations=state.get("citations") or [],
        conflicts=state.get("conflicts") or [],
    )

    return {
        "answer": result.get("answer", ""),
        "reasoning": result.get("reasoning"),
        "images_used": result.get("images_used") or [],
        "reasoning_effort_applied": result.get("reasoning_effort_applied", False),
    }


def _emit(state: dict[str, Any], msg: str) -> None:
    cb = state.get("status_callback")
    if cb is not None:
        try:
            cb(msg)
        except Exception:
            pass
