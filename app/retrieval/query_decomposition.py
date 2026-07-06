"""Low-fanout decomposition for compound v1 retrieval queries."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from app.core.ai_provider import invoke_llm_chat
from app.core.config import settings
from app.core.message_manager import extract_chat_response_text
from app.core.prompts import QUERY_PLANNER_DEVELOPER_PROMPT

MAX_DECOMPOSED_QUERIES = 4


class QueryDecompositionResponse(BaseModel):
    """Structured decomposition result returned by the query model."""

    queries: list[str] = Field(default_factory=list)


def decompose_query(
    question: str,
    max_queries: int = MAX_DECOMPOSED_QUERIES,
) -> list[str]:
    """Return up to ``max_queries`` standalone retrieval sub-queries."""
    normalized = " ".join(question.split())
    if not normalized:
        return []

    response = invoke_llm_chat(
        model=settings.V1_QUERY_ROUTER_MODEL or settings.QUERY_EXPANSION_MODEL,
        input_messages=[
            {"role": "developer", "content": QUERY_PLANNER_DEVELOPER_PROMPT},
            {
                "role": "user",
                "content": (
                    "Decompose the following question into retrieval sub-queries.\n\n"
                    f"Question:\n{normalized}"
                ),
            },
        ],
        reasoning_effort="none",
        max_output_tokens=300,
        prompt_cache_key="vectera:v1-query-decomposition:v1",
        prompt_cache_retention=settings.RESPONSE_PROMPT_CACHE_RETENTION,
        safety_identifier=(
            f"{settings.RESPONSE_SAFETY_IDENTIFIER_PREFIX}:v1-query-decomposition"
        ),
        user_tag=settings.RESPONSE_USER_TAG,
        timeout_seconds=25,
    )
    parsed = _parse_response(extract_chat_response_text(response))
    queries = _dedupe_queries(parsed.queries if parsed else [])
    if not queries:
        raise ValueError("Could not parse v1 query decomposition response")
    return queries[: max(1, max_queries)]


def _parse_response(content: str) -> QueryDecompositionResponse | None:
    cleaned = content.strip()
    if not cleaned:
        return None

    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            return QueryDecompositionResponse.model_validate_json(candidate)
        except Exception:
            try:
                return QueryDecompositionResponse.model_validate(json.loads(candidate))
            except Exception:
                continue
    return None


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = " ".join(str(query or "").split())
        normalized = cleaned.lower()
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(cleaned)
    return deduped
