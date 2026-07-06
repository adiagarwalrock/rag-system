"""
citation_builder_node: thin wrapper around build_citations().
"""

from __future__ import annotations

import logging
from typing import Any

from app.retrieval.citation_builder import build_citations

logger = logging.getLogger(__name__)


def citation_builder_node(state: dict[str, Any]) -> dict[str, Any]:
    evidence_nodes = state.get("evidence_nodes") or []

    citations = build_citations(evidence_nodes)

    logger.info("Citation builder produced %d citations", len(citations))

    return {"citations": citations}
