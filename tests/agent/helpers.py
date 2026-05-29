"""Shared test helpers for agent tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


def make_node(
    node_id: str = "node-1",
    score: float = 0.85,
    text: str = "Sample evidence text about FFO metrics.",
    metadata: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock LlamaIndex NodeWithScore."""
    node = MagicMock()
    node.node.node_id = node_id
    node.node.text = text
    node.node.metadata = metadata or {
        "section_summary": "This section covers FFO metrics.",
        "chunk_type": "text",
        "document_name": "Test Doc",
    }
    node.score = score
    return node


def make_state(**overrides: Any) -> dict[str, Any]:
    """Build a minimal AgentState dict for testing."""
    base: dict[str, Any] = {
        "question": "What is FFO per share?",
        "client_id": "client-1",
        "reasoning_effort": "medium",
        "reasoning_summary": None,
        "conversation_context": {},
        "status_callback": None,
        "reasoning_callback": None,
        "intent_labels": [],
        "route": "internal",
        "retrieved_nodes": [],
        "retrieved_node_ids": [],
        "evidence_nodes": [],
        "citations": [],
        "conflicts": [],
        "answer": "",
        "reasoning": None,
        "images_used": [],
        "reasoning_effort_applied": False,
        "iteration_count": 0,
        "evidence_sufficient": False,
        "retrieval_gap": None,
    }
    base.update(overrides)
    return base
