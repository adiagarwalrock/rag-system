"""
Query routing and low-fanout expansion for retrieval.
"""

import logging
from typing import List

from llama_index.core.prompts import PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.indexing.vector_store import is_placeholder_mode

logger = logging.getLogger(__name__)
QUERY_EXPANSION_MODEL = "gemini-3-flash-preview"
QUERY_EXPANSION_PROMPT = PromptTemplate(
    "Rewrite the user question into at most {max_rewrites} short retrieval queries "
    "for enterprise document RAG. Preserve concrete product names, dates, versions, "
    "and numeric terms. Do not answer the question.\n\n"
    "Return a structured object with `rewrites` containing only rewritten queries.\n\n"
    "Question: {question}"
)


class QueryRewriteResponse(BaseModel):
    rewrites: List[str] = Field(default_factory=list)


EXPANSION_TRIGGERS = (
    "compare",
    "comparison",
    "difference",
    "differences",
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

    Expansion is intentionally skipped for narrow lookup questions and when the
    app is running with mock LLM settings.
    """
    variants = [question.strip()]
    if not variants[0] or not should_expand_query(question):
        return variants

    if is_placeholder_mode():
        return variants

    try:
        llm = GoogleGenAI(
            model=QUERY_EXPANSION_MODEL,
            api_key=settings.google_api_key,
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
