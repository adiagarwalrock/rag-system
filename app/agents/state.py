"""
AgentState: shared state schema for the agentic RAG LangGraph pipeline.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── Inputs (set once at graph entry, never mutated) ──────────────────────
    question: str
    client_id: str
    reasoning_effort: str
    reasoning_summary: str | None
    conversation_context: dict[str, Any]
    status_callback: Any | None       # Callable[[str], None]
    reasoning_callback: Any | None    # Callable[[str], None]

    # ── Intent (set by intent_router_node) ───────────────────────────────────
    intent_labels: list[str]
    route: str                        # "internal" | "hybrid" (hybrid treated as internal for now)

    # ── Retrieval accumulation (operator.add appends across loop iterations) ─
    retrieved_nodes: Annotated[list[Any], operator.add]       # LlamaIndex NodeWithScore
    retrieved_node_ids: Annotated[list[str], operator.add]    # exclusion set for next pass

    # ── Post-loop (set once after loop exits) ────────────────────────────────
    evidence_nodes: list[Any]
    citations: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]

    # ── Answer ───────────────────────────────────────────────────────────────
    answer: str
    reasoning: str | None
    images_used: list[str]
    reasoning_effort_applied: bool

    # ── Loop control ─────────────────────────────────────────────────────────
    iteration_count: int
    evidence_sufficient: bool
    retrieval_gap: str | None         # reframe query hint from evidence_evaluator_node
