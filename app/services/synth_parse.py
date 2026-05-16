from __future__ import annotations

import json
import logging
import re
from typing import Any

from llama_index.core.base.llms.types import TextBlock, ThinkingBlock
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.core.structured_output import coerce_structured_output
from app.schemas.retrieval import (
    GroundedAnswerResult,
    GroundedAnswerStructuredResponse,
    format_reasoning_bullets,
)

logger = logging.getLogger(__name__)

_COMPLEX_QUERY_SIGNALS = (
    "what are the",
    "list all",
    "breakdown",
    "for each",
    "and what",
    "how much of",
    "what percentage",
    "quick facts",
    "fast facts",
    "headline stats",
)
_COMPLEX_QUERY_SIGNAL_THRESHOLD = 2
_VISUAL_SIGNALS = ("map", "chart", "graph", "figure", "diagram", "pie", "bar", "trend")
_JSON_BLOCK_PATTERN = re.compile(
    r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL
)
_THINKING_OPEN = re.compile(r"<thinking>", re.IGNORECASE)
_COT_ARTIFACT_PATTERN = re.compile(
    r"^(let me (think|analyze|break|consider)|step \d+[:.]|"
    r"first[,:]|to answer this|thinking:|analysis:)",
    re.IGNORECASE | re.MULTILINE,
)
_INLINE_CITATION_GROUP_PATTERN = re.compile(r"\[([0-9,\s]+)\]")
_TRUNCATION_SIGNALS = re.compile(r"(\.{3}$|\[trunc|\s\w{1,4}$)", re.IGNORECASE)


def effective_max_output_tokens(question: str) -> int:
    normalized = question.lower()
    visual_hit = any(t in normalized for t in _VISUAL_SIGNALS)
    signal_count = sum(1 for sig in _COMPLEX_QUERY_SIGNALS if sig in normalized)
    if visual_hit or signal_count >= _COMPLEX_QUERY_SIGNAL_THRESHOLD:
        return max(settings.RESPONSE_MAX_OUTPUT_TOKENS, 2000)
    return settings.RESPONSE_MAX_OUTPUT_TOKENS


def is_incomplete_structured_output_error(exc: Exception) -> bool:
    markers = (
        "invalid json: eof while parsing",
        "json_invalid",
        "failed to produce a structured response",
    )
    current: BaseException | None = exc
    while current is not None:
        message = str(current).lower()
        if any(marker in message for marker in markers):
            return True
        current = current.__cause__ or current.__context__
    return False


def validate_answer_completeness(answer: str) -> str:
    if not answer:
        return answer
    stripped = answer.rstrip()
    if _TRUNCATION_SIGNALS.search(stripped):
        logger.warning("Answer may be truncated: ends with %r", stripped[-30:])
    return answer


def to_grounded_answer_result(
    payload: GroundedAnswerStructuredResponse,
    *,
    image_paths: list[str],
    effort_applied: bool,
    evidence_count: int,
) -> GroundedAnswerResult:
    answer = normalize_inline_citations(payload.answer, evidence_count=evidence_count)
    return GroundedAnswerResult(
        answer=validate_answer_completeness(answer),
        reasoning=format_reasoning_bullets(payload.reasoning),
        images_used=image_paths,
        reasoning_effort_applied=effort_applied,
    )


def coerce_grounded_answer_structured_output(
    output: Any,
) -> GroundedAnswerStructuredResponse | None:
    coerced = coerce_structured_output(output, GroundedAnswerStructuredResponse)
    if coerced is not None:
        return coerced

    dict_candidates: list[dict[str, Any]] = []
    json_candidates: list[str] = []

    raw = getattr(output, "raw", None)
    if isinstance(raw, BaseModel):
        dict_candidates.append(raw.model_dump())
    if isinstance(output, BaseModel):
        dict_candidates.append(output.model_dump())
    if isinstance(raw, dict):
        dict_candidates.append(raw)
    if isinstance(output, dict):
        dict_candidates.append(output)

    message = getattr(output, "message", None)
    message_content = getattr(message, "content", None) if message is not None else None
    if isinstance(message_content, str) and message_content.strip():
        json_candidates.append(message_content)
    if isinstance(raw, str) and raw.strip():
        json_candidates.append(raw)
    if isinstance(output, str) and output.strip():
        json_candidates.append(output)

    for payload in dict_candidates:
        try:
            return GroundedAnswerStructuredResponse.model_validate(payload)
        except ValidationError:
            continue

    for payload in json_candidates:
        parsed = parse_grounded_answer_from_text(payload)
        if parsed is not None:
            return parsed

    return None


def normalize_inline_citations(answer: str, *, evidence_count: int) -> str:
    if not answer or evidence_count <= 0:
        return answer

    def _replace(match: re.Match[str]) -> str:
        raw_group = match.group(1)
        digits = [int(item) for item in re.findall(r"\d+", raw_group)]
        valid: list[int] = []
        seen: set[int] = set()
        for index in digits:
            if 1 <= index <= evidence_count and index not in seen:
                valid.append(index)
                seen.add(index)
        if not valid:
            return ""
        return "[" + ",".join(str(item) for item in valid) + "]"

    cleaned = _INLINE_CITATION_GROUP_PATTERN.sub(_replace, answer)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"(\[[0-9,]+\])\s+(\[[0-9,]+\])", r"\1\2", cleaned)
    return cleaned.strip()


def extract_answer_and_reasoning_from_chat(response: Any) -> tuple[str, str]:
    message = getattr(response, "message", None)
    if message is None:
        return extract_answer_and_reasoning_from_text(str(response).strip())

    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    for block in getattr(message, "blocks", None) or []:
        if isinstance(block, ThinkingBlock):
            content = (block.content or "").strip()
            if content:
                reasoning_parts.append(content)
            continue
        if isinstance(block, TextBlock):
            content = (block.text or "").strip()
            if content:
                answer_parts.append(content)

    answer_text = "\n".join(answer_parts).strip()
    reasoning_text = "\n\n".join(reasoning_parts).strip()
    fallback_text = answer_text or str(getattr(message, "content", "") or "").strip()
    parsed_answer, parsed_reasoning = extract_answer_and_reasoning_from_text(
        fallback_text
    )

    if answer_text and reasoning_text:
        return answer_text, reasoning_text
    if answer_text:
        return answer_text, parsed_reasoning
    return parsed_answer, reasoning_text or parsed_reasoning


def extract_answer_and_reasoning_from_text(text: str) -> tuple[str, str]:
    parsed = parse_grounded_answer_from_text(text)
    if parsed is not None:
        return parsed.answer, format_reasoning_bullets(parsed.reasoning)
    return split_reasoning_from_text(text)


def parse_grounded_answer_from_text(
    text: str,
) -> GroundedAnswerStructuredResponse | None:
    cleaned = text.strip()
    if not cleaned:
        return None

    payload = parse_grounded_answer_json(cleaned)
    if payload is not None:
        try:
            return GroundedAnswerStructuredResponse.model_validate(payload)
        except ValidationError:
            pass

        filtered_payload: dict[str, Any] = {}
        answer_value = payload.get("answer")
        if isinstance(answer_value, str):
            filtered_payload["answer"] = answer_value
        if "reasoning" in payload:
            filtered_payload["reasoning"] = payload.get("reasoning")
        if filtered_payload:
            try:
                return GroundedAnswerStructuredResponse.model_validate(filtered_payload)
            except ValidationError:
                return None
        return None

    answer, reasoning = split_reasoning_from_text(cleaned)
    if not answer.strip():
        return None
    fallback_payload: dict[str, Any] = {"answer": answer}
    if reasoning and reasoning.strip():
        fallback_payload["reasoning"] = [reasoning.strip()]
    try:
        return GroundedAnswerStructuredResponse.model_validate(fallback_payload)
    except ValidationError:
        return None


def parse_grounded_answer_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if not cleaned:
        return None

    candidate_blocks: list[str] = [cleaned]
    code_block_match = _JSON_BLOCK_PATTERN.match(cleaned)
    if code_block_match:
        candidate_blocks.append(code_block_match.group(1).strip())

    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        candidate_blocks.append(json_match.group(0))

    seen: set[str] = set()
    for block in candidate_blocks:
        candidate = block.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and (
            isinstance(parsed.get("answer"), str)
            or isinstance(parsed.get("reasoning"), str)
            or isinstance(parsed.get("reasoning"), list)
        ):
            return parsed

    return None


def split_reasoning_from_text(text: str) -> tuple[str, str]:
    """Return (answer, reasoning) with tolerant tag handling."""
    if not text:
        return "", ""

    cleaned = text.strip()
    thinking_m = re.search(
        r"<thinking>(.*?)</thinking>", cleaned, re.IGNORECASE | re.DOTALL
    )
    answer_m = re.search(r"<answer>(.*?)</answer>", cleaned, re.IGNORECASE | re.DOTALL)

    if thinking_m and answer_m:
        return _scrub_cot_artifacts(answer_m.group(1).strip()), thinking_m.group(1).strip()

    if answer_m:
        return _scrub_cot_artifacts(answer_m.group(1).strip()), ""

    if thinking_m:
        after_thinking = cleaned[thinking_m.end() :].strip()
        return _scrub_cot_artifacts(after_thinking), thinking_m.group(1).strip()

    open_m = _THINKING_OPEN.search(cleaned)
    if open_m:
        before = cleaned[: open_m.start()].strip()
        if before:
            return _scrub_cot_artifacts(before), ""
        return "", cleaned[open_m.end() :].strip()

    return _scrub_cot_artifacts(cleaned), ""


def scrub_cot_artifacts(text: str) -> str:
    return _scrub_cot_artifacts(text)


def _scrub_cot_artifacts(text: str) -> str:
    lines = text.splitlines()
    clean = []
    for line in lines:
        if _COT_ARTIFACT_PATTERN.match(line.strip()):
            continue
        clean.append(line)
    return "\n".join(clean).strip()


def build_source_grounded_fallback(citations: list[dict[str, Any]]) -> str:
    labels = []
    for citation in citations[:3]:
        label = citation.get("citation_label") or citation.get("document_name")
        if label:
            labels.append(label)

    if labels:
        return (
            "I found relevant evidence, but could not synthesize a final answer. "
            f"Review these sources: {', '.join(labels)}."
        )
    return "I found relevant evidence, but could not synthesize a final answer."
