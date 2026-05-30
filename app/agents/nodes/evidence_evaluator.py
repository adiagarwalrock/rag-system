"""
evidence_evaluator_node: decides whether accumulated retrieved_nodes are sufficient
to answer the question, or whether another retrieval pass is needed.

Uses QUERY_EXPANSION_MODEL (fast mini) for the binary judgment to keep latency low.
Forces evidence_sufficient=True when iteration_count >= AGENTIC_MAX_ITERATIONS.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from app.agents.nodes._shared import emit_status
from app.core.ai_provider import invoke_llm_chat
from app.core.config import settings

logger = logging.getLogger(__name__)


class EvidenceEvaluation(BaseModel):
    sufficient: bool
    gap: str | None
    node_scores: list[bool]  # one bool per sampled node — True = directly addresses question


_SYSTEM_PROMPT = """\
You are an evidence sufficiency evaluator for a financial document RAG system.

Given a question and sampled evidence chunks, decide whether the evidence is sufficient
to produce a well-grounded answer.

Rules:
- For comparison questions (A vs B, compare X and Y, how does X differ from Y): evidence is
  ONLY sufficient if chunks covering EACH named entity are present with relevant financial content.
  If one entity is missing or only represented by cover/appendix/table-of-contents pages,
  set sufficient=false and name both entities in the gap field.
- For single-entity questions: sufficient=true if the key metric or fact appears in at least
  one chunk with substantive financial data.
- If the evidence consists entirely of cover pages, table-of-contents pages, or appendix headings
  with no substantive financial data, set sufficient=false and set gap to:
  'No substantive financial content retrieved — only cover or appendix pages.'
- Do not infer coverage from tangential mentions — the chunk must directly address the entity
  or metric with specific financial data.

For each sampled evidence chunk, set node_scores[i]=true if that chunk directly addresses the
question with specific financial data, false otherwise.

Return a JSON object with:
- sufficient: true or false
- gap: what is missing (max 20 words), or null if sufficient
- node_scores: array of booleans, one per sampled chunk in the order listed
"""


def evidence_evaluator_node(state: dict[str, Any]) -> dict[str, Any]:
    iteration_count = state.get("iteration_count", 0)
    max_iterations = settings.AGENTIC_MAX_ITERATIONS

    # Hard guard — force synthesis after max iterations regardless of evidence quality
    if iteration_count >= max_iterations:
        logger.info("Agentic max iterations (%d) reached; forcing synthesis", max_iterations)
        return {"evidence_sufficient": True, "retrieval_gap": None}

    retrieved_nodes = state.get("retrieved_nodes") or []
    if not retrieved_nodes:
        return {"evidence_sufficient": False, "retrieval_gap": f"general context about: {state['question'][:80]}"}

    emit_status(state, "Evaluating evidence sufficiency…")

    # Deterministic coverage check before LLM call — catches comparison questions where
    # one entity's nodes are present but low-relevance (e.g. appendix/cover pages).
    det_sufficient, det_gap = _coverage_check(state["question"], retrieved_nodes)
    if not det_sufficient:
        logger.info(
            "Evidence evaluator (iter=%d): deterministic gap detected: %r",
            iteration_count,
            det_gap,
        )
        return {"evidence_sufficient": False, "retrieval_gap": det_gap}

    sufficient, gap = _llm_evaluate(
        question=state["question"],
        nodes=retrieved_nodes,
    )
    logger.info(
        "Evidence evaluator (iter=%d): sufficient=%s gap=%r",
        iteration_count,
        sufficient,
        gap,
    )
    return {"evidence_sufficient": sufficient, "retrieval_gap": gap}


_COMPARISON_SIGNALS = (
    "vs",
    "versus",
    "compare",
    "compared",
    "differ",
    "difference",
    "both",
    "and",
    "between",
)
_MIN_ENTITY_HITS = 3  # nodes whose text must mention the entity


def _coverage_check(question: str, nodes: list[Any]) -> tuple[bool, str | None]:
    """Deterministic check: for comparison questions, verify each entity has ≥ MIN hits."""
    q_lower = question.lower()
    if not any(sig in q_lower for sig in _COMPARISON_SIGNALS):
        return True, None  # not a comparison — let LLM decide

    # Extract candidate entity names: words/phrases ≥ 3 chars starting with uppercase
    import re

    # Grab runs of title-case words (company names like "Realty Income", "VICI")
    entities = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\b", question)
    # Filter out short stop-words and question words
    skip = {"How", "What", "Both", "Their", "The", "Are", "Do", "Does", "Which", "And"}
    entities = [e for e in entities if e not in skip and len(e) >= 3]
    if len(entities) < 2:
        return True, None  # can't identify two sides — let LLM decide

    # Count text hits per entity across all nodes
    all_text = [
        (n.node.text or "").lower()
        + " "
        + str((n.node.metadata or {}).get("section_summary", "")).lower()
        for n in nodes
    ]
    entity_hits: dict[str, int] = {}
    for entity in entities:
        ent_lower = entity.lower()
        entity_hits[entity] = sum(1 for t in all_text if ent_lower in t)

    # Find the weakest-covered entity
    min_entity = min(entity_hits, key=lambda e: entity_hits[e])
    min_hits = entity_hits[min_entity]
    logger.debug("Coverage check entity hits: %s", entity_hits)

    if min_hits < _MIN_ENTITY_HITS:
        # Extract topic keywords from question (non-entity words, ≥4 chars)
        entity_words = {w.lower() for e in entities for w in e.split()}
        topic_words = [
            w
            for w in re.findall(r"\b[a-z]{4,}\b", question.lower())
            if w not in entity_words
            and w
            not in {
                "what",
                "their",
                "both",
                "does",
                "have",
                "from",
                "with",
                "that",
                "this",
            }
        ]
        topic = " ".join(topic_words[:4]) if topic_words else "portfolio details"
        gap = f"{min_entity} {topic}"
        return False, gap

    return True, None


def _llm_evaluate(question: str, nodes: list[Any]) -> tuple[bool, str | None]:
    model: str = (
        settings.AGENTIC_EVIDENCE_EVALUATOR_MODEL
        or settings.QUERY_EXPANSION_MODEL
        or settings.LLM_MODEL
    )

    sorted_nodes = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
    # Top-2 by score + up to 3 evenly-spaced samples from the rest for coverage breadth
    sampled = sorted_nodes[:2]
    tail = sorted_nodes[2:]
    if tail:
        step = max(1, len(tail) // 3)
        sampled += tail[::step][:3]
    evidence_lines = []
    for i, node in enumerate(sampled, start=1):
        node_id = getattr(node.node, "node_id", "unknown")
        score = node.score or 0.0
        summary = (node.node.metadata or {}).get("section_summary", "")
        preview = (node.node.text or "")[:200].replace("\n", " ")
        evidence_lines.append(
            f"[{i}] id={node_id} score={score:.3f}\n"
            f"    summary: {summary}\n"
            f"    preview: {preview}"
        )

    # Best chunk per source — gives the LLM one representative line per document
    best_by_source: dict[str, Any] = {}
    for n in nodes:
        fname = (
            (n.node.metadata or {}).get("file_name")
            or (n.node.metadata or {}).get("source")
            or "unknown"
        )
        if fname not in best_by_source or (n.score or 0) > (
            best_by_source[fname].score or 0
        ):
            best_by_source[fname] = n
    source_coverage_lines = []
    for fname, n in sorted(best_by_source.items()):
        summary = (n.node.metadata or {}).get("section_summary", "") or (
            n.node.text or ""
        )[:80]
        source_coverage_lines.append(
            f"  {fname}: best_score={n.score:.3f} — {summary[:120]}"
        )

    user_content = f"Question: {question}\n\n" f"Coverage across all {len(nodes)} retrieved chunks ({len(best_by_source)} source documents):\n" + "\n".join(
        source_coverage_lines
    ) + "\n\nSample evidence (top-2 by score + 3 breadth samples):\n" + "\n".join(
        evidence_lines
    )

    try:
        result: EvidenceEvaluation = invoke_llm_chat(
            model=model,
            input_messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_output_tokens=256,
            timeout_seconds=15.0,
            structured_output_schema=EvidenceEvaluation,
        )
        logger.debug(
            "LLM evaluation: sufficient=%s gap=%r node_scores=%s",
            result.sufficient,
            result.gap,
            result.node_scores,
        )
        return result.sufficient, result.gap or None
    except Exception:
        logger.warning("Evidence evaluator LLM call failed; defaulting to sufficient=True", exc_info=True)
        return True, None


