"""
Query routing and low-fanout expansion for retrieval.
"""

import logging
import re

from llama_index.core import Settings

from app.indexing.vector_store import is_placeholder_mode

logger = logging.getLogger(__name__)

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
        prompt = (
            "Rewrite the user question into at most "
            f"{max_rewrites} short retrieval queries for enterprise document RAG. "
            "Preserve concrete product names, dates, versions, and numeric terms. "
            "Do not answer the question. Return one query per line.\n\n"
            f"Question: {question}"
        )
        response = Settings.llm.complete(prompt)
    except Exception:
        logger.exception("Query expansion failed; using original question only")
        return variants

    for line in str(response).splitlines():
        rewrite = _clean_rewrite(line)
        if not rewrite or rewrite.lower() == variants[0].lower():
            continue
        if rewrite.lower() in {variant.lower() for variant in variants}:
            continue
        variants.append(rewrite)
        if len(variants) >= max_rewrites + 1:
            break

    return variants


def _clean_rewrite(line: str) -> str:
    rewrite = line.strip()
    rewrite = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", rewrite)
    return rewrite.strip().strip('"')
