"""
Query routing and low-fanout expansion for retrieval.
"""

import json
import logging
import re
from typing import Any, List

from pydantic import BaseModel, Field

from app.core.ai_provider import extract_chat_response_text, invoke_llm_chat
from app.core.config import settings
from app.core.prompts import QUERY_EXPANSION_DEVELOPER_PROMPT

logger = logging.getLogger(__name__)


class QueryRewriteResponse(BaseModel):
    rewrites: List[str] = Field(default_factory=list)


MAX_HISTORY_TURNS = 6
MAX_HISTORY_TURN_CHARS = 220


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

FOLLOW_UP_PHRASES = (
    "what about",
    "how about",
    "compare that",
    "compare it",
    "vs that",
    "versus that",
    "previous one",
    "earlier one",
    "same one",
    "that one",
)

FOLLOW_UP_TOKENS = {
    "it",
    "its",
    "that",
    "those",
    "this",
    "these",
    "they",
    "them",
    "previous",
    "earlier",
    "prior",
    "same",
    "above",
    "latter",
    "former",
}


def _normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate_with_ellipsis(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 3].rstrip()}..."


def _normalize_recent_turns(raw_turns: Any) -> list[dict[str, str]]:
    turns = raw_turns if isinstance(raw_turns, list) else []
    normalized: list[dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _normalize_space(turn.get("content"))
        if not content:
            continue
        normalized.append(
            {
                "role": role,
                "content": _truncate_with_ellipsis(content, MAX_HISTORY_TURN_CHARS),
            }
        )
    return normalized[-MAX_HISTORY_TURNS:]


def _parse_query_input(
    question: str | dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    if isinstance(question, dict):
        current_question = _normalize_space(question.get("current_question"))
        if not current_question:
            current_question = _normalize_space(question.get("question"))
        if not current_question:
            current_question = _normalize_space(question.get("text"))
        recent_turns = _normalize_recent_turns(question.get("recent_turns"))
        return current_question, recent_turns
    return _normalize_space(question), []


def _is_contextual_follow_up(question_text: str) -> bool:
    normalized = question_text.lower()
    if any(phrase in normalized for phrase in FOLLOW_UP_PHRASES):
        return True

    tokens = re.findall(r"[a-z0-9']+", normalized)
    if not tokens:
        return False
    return len(tokens) <= 10 and any(token in FOLLOW_UP_TOKENS for token in tokens)


def should_expand_query(question: str | dict[str, Any]) -> bool:
    """Return True for broad, comparative, versioned, or conflict-oriented queries."""
    current_question, recent_turns = _parse_query_input(question)
    normalized = current_question.lower()
    if any(trigger in normalized for trigger in EXPANSION_TRIGGERS):
        return True
    return bool(recent_turns) and _is_contextual_follow_up(current_question)


def build_query_variants(
    question: str | dict[str, Any], max_rewrites: int = 2
) -> list[str]:
    """
    Return the original question plus up to max_rewrites retrieval rewrites.

    Expansion is intentionally skipped for narrow lookup questions.
    """
    current_question, recent_turns = _parse_query_input(question)
    variants = [current_question]
    if not variants[0] or not should_expand_query(question):
        return variants

    try:
        response = _predict_query_rewrites(
            current_question=current_question,
            recent_turns=recent_turns,
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


def _predict_query_rewrites(
    *,
    current_question: str,
    recent_turns: list[dict[str, str]],
    max_rewrites: int,
) -> QueryRewriteResponse:
    response = invoke_llm_chat(
        model=settings.QUERY_EXPANSION_MODEL,
        input_messages=_build_query_expansion_messages(
            current_question=current_question,
            recent_turns=recent_turns,
            max_rewrites=max_rewrites,
        ),
        reasoning_effort="low",
        max_output_tokens=220,
        prompt_cache_key="rag:query-expansion:v2",
        prompt_cache_retention=settings.RESPONSE_PROMPT_CACHE_RETENTION,
        safety_identifier=(
            f"{settings.RESPONSE_SAFETY_IDENTIFIER_PREFIX}:query-expansion"
        ),
        user_tag=settings.RESPONSE_USER_TAG,
    )
    content = extract_chat_response_text(response)
    parsed = _parse_query_rewrite_response(content)
    if parsed is None:
        raise ValueError("Could not parse query expansion response")
    return parsed


def _build_query_expansion_messages(
    *,
    current_question: str,
    recent_turns: list[dict[str, str]],
    max_rewrites: int,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {
            "role": "developer",
            "content": QUERY_EXPANSION_DEVELOPER_PROMPT,
        }
    ]

    for turn in recent_turns:
        messages.append(
            {
                "role": turn["role"],
                "content": turn["content"],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": (
                "Rewrite the latest user question into retrieval queries. "
                "Use prior turns only for reference resolution. "
                "Do not answer.\n\n"
                f"Latest question:\n{current_question}\n\n"
                f"max_rewrites={max_rewrites}"
            ),
        }
    )
    return messages


def _parse_query_rewrite_response(content: str) -> QueryRewriteResponse | None:
    cleaned = content.strip()
    if not cleaned:
        return None

    candidate_blocks = [cleaned]
    if cleaned.startswith("```"):
        stripped = cleaned.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        candidate_blocks.append(stripped)

    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        candidate_blocks.append(json_match.group(0))

    for block in candidate_blocks:
        try:
            return QueryRewriteResponse.model_validate_json(block)
        except Exception:
            pass

    try:
        payload = json.loads(cleaned)
        return QueryRewriteResponse.model_validate(payload)
    except Exception:
        return None
