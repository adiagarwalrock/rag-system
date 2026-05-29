"""
conflict_detector_node: thin wrapper around detect_conflicts().
"""

from __future__ import annotations

import logging
from typing import Any

from app.retrieval.conflict_detector import detect_conflicts

logger = logging.getLogger(__name__)


def conflict_detector_node(state: dict[str, Any]) -> dict[str, Any]:
    retrieved_nodes = state.get("retrieved_nodes") or []
    evidence_nodes = state.get("evidence_nodes") or []
    question = state["question"]

    conflicts = detect_conflicts(
        source_nodes=retrieved_nodes,
        evidence_nodes=evidence_nodes,
        question=question,
    )

    if conflicts:
        logger.info("Conflict detector found %d conflict(s)", len(conflicts))

    return {"conflicts": conflicts}
