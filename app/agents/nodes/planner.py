"""
planner_node: decomposes the user question into an ordered list of retrieval sub-queries.

Runs once before the first vector_retrieval pass. On failure falls back to [question]
so the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agents.nodes._shared import emit_status
from app.core.ai_provider import invoke_llm_chat
from app.core.message_manager import extract_chat_response_text
from app.core.config import settings
from app.core.prompts import QUERY_PLANNER_DEVELOPER_PROMPT

logger = logging.getLogger(__name__)

_MAX_QUERIES = 4

#TODO: Use Structured responses with pydantic base
# classses and clear valudation and descriptions

def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    question = state["question"]
    emit_status(state, "Planning retrieval sub-queries…")

    try:
        planned = _predict_queries(question)
    except Exception:
        logger.exception("Planner node failed; falling back to original question")
        planned = [question]

    logger.info("Planner produced %d sub-queries: %s", len(planned), planned)
    return {"planned_queries": planned}


def _predict_queries(question: str) -> list[str]:
    model = settings.AGENTIC_PLANNER_MODEL or settings.QUERY_EXPANSION_MODEL
    response = invoke_llm_chat(
        model=model,
        input_messages=[
            {"role": "developer", "content": QUERY_PLANNER_DEVELOPER_PROMPT},
            {
                "role": "user",
                "content": (
                    "Decompose the following question into retrieval sub-queries.\n\n"
                    f"Question:\n{question}"
                ),
            },
        ],
        reasoning_effort="none",
        max_output_tokens=300,
        prompt_cache_key="vectera:query-planner:v1",
        prompt_cache_retention=settings.RESPONSE_PROMPT_CACHE_RETENTION,
        safety_identifier=(
            f"{settings.RESPONSE_SAFETY_IDENTIFIER_PREFIX}:query-planner"
        ),
        user_tag=settings.RESPONSE_USER_TAG,
        timeout_seconds=25,
    )
    content = extract_chat_response_text(response)
    queries = _parse_response(content)
    if not queries:
        raise ValueError("Could not parse planner response")
    return queries[:_MAX_QUERIES]


def _parse_response(content: str) -> list[str]:
    cleaned = content.strip()
    if not cleaned:
        return []

    def _extract(block: str) -> list[str]:
        try:
            payload = json.loads(block)
            queries = payload.get("queries") or []
            return [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        except Exception:
            return []

    result = _extract(cleaned)
    if result:
        return result

    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        return _extract(json_match.group(0))

    return []


