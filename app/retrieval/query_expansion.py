"""
Query routing and low-fanout expansion for retrieval.
"""

import logging
from typing import List

from pydantic import BaseModel, Field

from app.core.ai_provider import get_llm
from app.core.config import settings
from app.core.prompts import QUERY_EXPANSION_PROMPT

logger = logging.getLogger(__name__)


class QueryRewriteResponse(BaseModel):
    rewrites: List[str] = Field(default_factory=list)


EXPANSION_TRIGGERS = (
    "compare",
    "comparison",
    "difference",
    "differences",
    "change",
    "changed",
    "changes",
    "current version",
    "older version",
    "previous version",
    "latest",
    "conflict",
    "conflicting",
    "contradict",
    "trend",
    "trends",
    "summarize",
    "summary",
    "across",
    "over time",
    "chart",
    "charts",
    "graph",
    "graphs",
    "map",
    "maps",
    "table",
    "tables",
    "figure",
    "figures",
    "diagram",
    "image",
    "screenshot",
)


def should_expand_query(question: str) -> bool:
    """Return True for broad, comparative, versioned, or conflict-oriented queries."""
    normalized = question.lower()
    return any(trigger in normalized for trigger in EXPANSION_TRIGGERS)


def build_query_variants(question: str, max_rewrites: int = 2) -> list[str]:
    """
    Return the original question plus up to max_rewrites retrieval rewrites.

    Expansion is intentionally skipped for narrow lookup questions.
    """
    variants = [question.strip()]
    if not variants[0] or not should_expand_query(question):
        return variants

    try:
        llm = get_llm(
            model=settings.QUERY_EXPANSION_MODEL,
            api_key=settings.ai_api_key,
        )
        response = llm.structured_predict(
            QueryRewriteResponse,
            QUERY_EXPANSION_PROMPT,
            question=question,
            max_rewrites=max_rewrites,
        )
    except Exception:
        logger.exception("Query expansion failed; using original question only")
        return variants

    for rewrite_candidate in response.rewrites:
        rewrite = rewrite_candidate.strip()
        if not rewrite or rewrite.lower() == variants[0].lower():
            continue
        if rewrite.lower() in {variant.lower() for variant in variants}:
            continue
        variants.append(rewrite)
        if len(variants) >= max_rewrites + 1:
            break

    return variants
