"""
intent_router_node: classifies the query as "internal" or "hybrid" using an LLM.

"hybrid" is stored in state for future use (e.g. stricter evidence evaluation)
but is routed identically to "internal" in the current graph.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.ai_provider import invoke_llm_chat
from app.core.config import settings
from app.retrieval.query_intent import analyze_retrieval_intent

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a query router for a financial document RAG system.

The internal corpus contains: uploaded investor presentations, earnings transcripts,
annual reports, supplemental filings, and strategic documents from specific REITs and
real-estate companies.

Classify the user query into exactly one route:
- "internal"  → fully answerable from the uploaded financial documents
- "hybrid"    → references a document topic but also requires external context
                (current market rates, recent news, sector benchmarks, macro data)

Respond with JSON only — no prose, no markdown fences:
{"route": "internal" or "hybrid", "reason": "<10 words max>"}
"""


def intent_router_node(state: dict[str, Any]) -> dict[str, Any]:
    question = state["question"]
    _emit(state, "Classifying query intent…")

    existing_intent = analyze_retrieval_intent(question)
    route = _llm_classify(question)

    return {
        "intent_labels": list(existing_intent.labels),
        "route": route,
    }


def _llm_classify(question: str) -> str:
    model = settings.QUERY_EXPANSION_MODEL or settings.LLM_MODEL
    try:
        response = invoke_llm_chat(
            model=model,
            input_messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_output_tokens=64,
            timeout_seconds=10.0,
        )
        text = str(response.message.content or "").strip()
        parsed = json.loads(text)
        route = parsed.get("route", "internal")
        if route not in ("internal", "hybrid"):
            return "internal"
        return route
    except Exception:
        logger.warning("Intent router LLM call failed; defaulting to 'internal'", exc_info=True)
        return "internal"


def _emit(state: dict[str, Any], msg: str) -> None:
    cb = state.get("status_callback")
    if cb is not None:
        try:
            cb(msg)
        except Exception:
            pass
