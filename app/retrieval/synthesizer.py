"""
Grounded answer synthesis via LLM.

Extracted from retriever.py to isolate the LLM-backed answer generation
domain from retrieval orchestration.
"""

import json
import logging
import mimetypes
import re
from typing import Any
from pathlib import Path

from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
    ThinkingBlock,
)
from pydantic import BaseModel, ValidationError

from app.core.ai_provider import get_llm
from app.core.config import settings
from app.core.prompts import GROUNDED_ANSWER_DEVELOPER_PROMPT
from app.schemas.retrieval import (
    GroundedAnswerResult,
    GroundedAnswerStructuredResponse,
    format_reasoning_bullets,
)
from app.retrieval.prompt_builder import (
    _build_grounded_prompt,
    _build_labeled_context_sections,
    _collect_image_evidence_paths,
)

logger = logging.getLogger(__name__)

# Signals that indicate a question has multiple sub-parts requiring a longer answer.
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
_JSON_BLOCK_PATTERN = re.compile(
    r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL
)


def _effective_max_output_tokens(question: str) -> int:
    """Return a larger token budget for multi-part enumeration questions."""
    normalized = question.lower()
    signal_count = sum(1 for sig in _COMPLEX_QUERY_SIGNALS if sig in normalized)
    if signal_count >= _COMPLEX_QUERY_SIGNAL_THRESHOLD:
        return max(settings.RESPONSE_MAX_OUTPUT_TOKENS, 2000)
    return settings.RESPONSE_MAX_OUTPUT_TOKENS


def _is_incomplete_structured_output_error(exc: Exception) -> bool:
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


def _to_grounded_answer_result(
    payload: GroundedAnswerStructuredResponse,
    *,
    image_paths: list[str],
    effort_applied: bool,
) -> "GroundedAnswerResult":
    return GroundedAnswerResult(
        answer=payload.answer,
        reasoning=format_reasoning_bullets(payload.reasoning),
        images_used=image_paths,
        reasoning_effort_applied=effort_applied,
    )


def _coerce_grounded_answer_structured_output(
    output: Any,
) -> GroundedAnswerStructuredResponse | None:
    if isinstance(output, GroundedAnswerStructuredResponse):
        return output

    raw = getattr(output, "raw", None)
    if isinstance(raw, GroundedAnswerStructuredResponse):
        return raw

    dict_candidates: list[dict[str, Any]] = []
    json_candidates: list[str] = []

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
        parsed = _parse_grounded_answer_from_text(payload)
        if parsed is not None:
            return parsed

    return None


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class GroundedAnswerSynthesizer:
    """Synthesize grounded answers from selected evidence and conflicts."""

    def __init__(
        self,
        *,
        client_id: str,
        reasoning_effort: str,
        conversation_context: dict[str, Any],
    ):
        self.client_id = client_id
        self.reasoning_effort = reasoning_effort
        self.conversation_context = conversation_context

    def synthesize(
        self,
        *,
        question: str,
        citations: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        effort_applied = settings.OPENAI_USE_RESPONSES

        if not citations:
            return GroundedAnswerResult(
                answer=(
                    "I could not find enough relevant evidence in the uploaded "
                    "documents to answer this question confidently."
                ),
                reasoning="",
                images_used=[],
                reasoning_effort_applied=effort_applied,
            ).to_dict()

        image_paths, citation_image_map = _collect_image_evidence_paths(citations)

        if settings.OPENAI_USE_RESPONSES:
            responses_result = self._try_responses_synthesis(
                question=question,
                citations=citations,
                conflicts=conflicts,
                image_paths=image_paths,
                citation_image_map=citation_image_map,
                effort_applied=effort_applied,
            )
            if responses_result is not None:
                return responses_result.to_dict()

        chat_result = self._try_chat_synthesis(
            question=question,
            citations=citations,
            conflicts=conflicts,
            image_paths=image_paths,
            citation_image_map=citation_image_map,
            effort_applied=effort_applied,
        )
        if chat_result is not None:
            return chat_result.to_dict()

        logger.error("Answer synthesis failed; returning source-grounded fallback")
        return GroundedAnswerResult(
            answer=_build_source_grounded_fallback(citations),
            reasoning="",
            images_used=image_paths,
            reasoning_effort_applied=effort_applied,
        ).to_dict()

    def _try_responses_synthesis(
        self,
        *,
        question: str,
        citations: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
        image_paths: list[str],
        citation_image_map: dict[int, list[int]],
        effort_applied: bool,
    ) -> GroundedAnswerResult | None:
        try:
            import app.retrieval.retriever as _retriever_mod

            budgeter = _retriever_mod.ResponsesInputBudgeter(model=settings.LLM_MODEL)
            sections = _build_labeled_context_sections(
                citations=citations,
                conflicts=conflicts,
                conversation_context=self.conversation_context,
            )
            input_messages, _, metrics = budgeter.build_budgeted_sections(
                developer_prompt=GROUNDED_ANSWER_DEVELOPER_PROMPT,
                question=question,
                recent_turns=self.conversation_context.get("recent_turns") or [],
                session_summary=sections["session_summary"],
                cross_session_lines=sections["cross_session_lines"],
                evidence_lines=sections["evidence_lines"],
                conflict_lines=sections["conflict_lines"],
            )
            logger.info(
                "Responses budget usage model=%s input_tokens=%d/%d history=%d summary=%d cross=%d evidence=%d conflict=%d",
                metrics.model,
                metrics.total_input_tokens,
                metrics.input_budget_tokens,
                metrics.history_tokens,
                metrics.summary_tokens,
                metrics.cross_session_tokens,
                metrics.evidence_tokens,
                metrics.conflict_tokens,
            )

            if image_paths:
                input_messages = _append_image_inputs(
                    input_messages=input_messages,
                    image_paths=image_paths,
                )

            request_kwargs = {
                "model": settings.LLM_MODEL,
                "input_messages": input_messages,
                "reasoning_effort": self.reasoning_effort,
                "max_output_tokens": _effective_max_output_tokens(question),
                "prompt_cache_key": settings.RESPONSE_PROMPT_CACHE_KEY,
                "prompt_cache_retention": settings.RESPONSE_PROMPT_CACHE_RETENTION,
                "safety_identifier": f"{settings.RESPONSE_SAFETY_IDENTIFIER_PREFIX}:{self.client_id}",
                "user_tag": settings.RESPONSE_USER_TAG,
            }

            try:
                structured_response = _retriever_mod.invoke_llm_chat(
                    model=settings.LLM_MODEL,
                    input_messages=input_messages,
                    reasoning_effort=self.reasoning_effort,
                    max_output_tokens=_effective_max_output_tokens(question),
                    prompt_cache_key=settings.RESPONSE_PROMPT_CACHE_KEY,
                    prompt_cache_retention=settings.RESPONSE_PROMPT_CACHE_RETENTION,
                    safety_identifier=f"{settings.RESPONSE_SAFETY_IDENTIFIER_PREFIX}:{self.client_id}",
                    user_tag=settings.RESPONSE_USER_TAG,
                    structured_output_cls=GroundedAnswerStructuredResponse,
                )
            except Exception as exc:
                if _is_incomplete_structured_output_error(exc):
                    logger.warning(
                        "Structured grounded answer output incomplete; retrying with parser fallback"
                    )
                else:
                    logger.exception(
                        "Structured grounded answer synthesis failed; retrying with parser fallback"
                    )

            if structured_response is not None:
                structured_payload = _coerce_grounded_answer_structured_output(
                    structured_response
                )
                if structured_payload is not None:
                    return _to_grounded_answer_result(
                        structured_payload,
                        image_paths=image_paths,
                        effort_applied=effort_applied,
                    )

                logger.warning(
                    "Structured grounded answer payload invalid; using parser fallback on response text"
                )
                structured_fallback = _coerce_grounded_answer_structured_output(
                    _retriever_mod.extract_chat_response_text(structured_response)
                )
                if structured_fallback is not None:
                    return _to_grounded_answer_result(
                        structured_fallback,
                        image_paths=image_paths,
                        effort_applied=effort_applied,
                    )

            fallback_response = _retriever_mod.invoke_llm_chat(
                model=settings.LLM_MODEL,
                input_messages=input_messages,
                reasoning_effort=self.reasoning_effort,
                max_output_tokens=_effective_max_output_tokens(question),
                prompt_cache_key=settings.RESPONSE_PROMPT_CACHE_KEY,
                prompt_cache_retention=settings.RESPONSE_PROMPT_CACHE_RETENTION,
                safety_identifier=f"{settings.RESPONSE_SAFETY_IDENTIFIER_PREFIX}:{self.client_id}",
                user_tag=settings.RESPONSE_USER_TAG,
            )
            parsed_fallback = _parse_grounded_answer_from_text(
                _retriever_mod.extract_chat_response_text(fallback_response)
            )
            if parsed_fallback is None:
                raise ValueError("LLM returned empty answer")
            return _to_grounded_answer_result(
                parsed_fallback,
                image_paths=image_paths,
                effort_applied=effort_applied,
            )
        except Exception:
            logger.exception(
                "Responses answer synthesis failed; falling back to LlamaIndex chat"
            )
            return None

    def _try_chat_synthesis(
        self,
        *,
        question: str,
        citations: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
        image_paths: list[str],
        citation_image_map: dict[int, list[int]],
        effort_applied: bool,
    ) -> GroundedAnswerResult | None:
        llm = get_llm(reasoning_effort=self.reasoning_effort)
        prompt = _build_grounded_prompt(
            question,
            citations,
            conflicts,
            image_attachment_count=len(image_paths),
            conversation_context=self.conversation_context,
            citation_image_map=citation_image_map,
        )

        for attempt_name, message, used_images in self._chat_attempts(
            prompt, image_paths
        ):
            try:
                response = llm.chat([message])
                answer, reasoning = _extract_answer_and_reasoning_from_chat(response)
                try:
                    parsed_chat = GroundedAnswerStructuredResponse.model_validate(
                        {
                            "answer": answer,
                            "reasoning": [reasoning] if reasoning else [],
                        }
                    )
                except ValidationError as exc:
                    raise ValueError("LLM returned empty answer") from exc
                return _to_grounded_answer_result(
                    parsed_chat,
                    image_paths=used_images,
                    effort_applied=effort_applied,
                )
            except Exception:
                logger.exception("%s answer synthesis failed", attempt_name)

        return None

    @staticmethod
    def _chat_attempts(
        prompt: str,
        image_paths: list[str],
    ) -> list[tuple[str, ChatMessage, list[str]]]:
        attempts: list[tuple[str, ChatMessage, list[str]]] = []
        if image_paths:
            attempts.append(
                (
                    "multimodal",
                    _build_grounded_message(prompt, image_paths),
                    image_paths,
                )
            )
        attempts.append(("text_only", _build_grounded_message(prompt, []), []))
        return attempts


# ---------------------------------------------------------------------------
# LLM message helpers
# ---------------------------------------------------------------------------


def _build_grounded_message(prompt: str, image_paths: list[str]) -> ChatMessage:
    blocks: list[Any] = [TextBlock(text=prompt)]
    for image_path in image_paths:
        mime_type = mimetypes.guess_type(image_path)[0]
        blocks.append(
            ImageBlock(
                path=Path(image_path),
                image_mimetype=mime_type or "image/png",
            )
        )
    return ChatMessage(role=MessageRole.USER, blocks=blocks)


def _append_image_inputs(
    *,
    input_messages: list[dict[str, Any]],
    image_paths: list[str],
) -> list[dict[str, Any]]:
    if not image_paths:
        return input_messages

    messages = [*input_messages]
    if not messages:
        return messages

    user_message = messages[-1]
    if user_message.get("role") != "user":
        return messages

    user_text = str(user_message.get("content") or "")
    multimodal_content: list[dict[str, Any]] = [
        {"type": "input_text", "text": user_text}
    ]
    for path in image_paths:
        from app.retrieval.prompt_builder import _image_path_to_data_url

        data_url = _image_path_to_data_url(path)
        if not data_url:
            continue
        multimodal_content.append({"type": "input_image", "image_url": data_url})

    user_message["content"] = multimodal_content
    return messages


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_answer_and_reasoning_from_chat(response: Any) -> tuple[str, str]:
    message = getattr(response, "message", None)
    if message is None:
        return _extract_answer_and_reasoning_from_text(str(response).strip())

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
    parsed_answer, parsed_reasoning = _extract_answer_and_reasoning_from_text(
        fallback_text
    )

    if reasoning_text:
        return parsed_answer, reasoning_text
    return parsed_answer, parsed_reasoning


def _extract_answer_and_reasoning_from_text(text: str) -> tuple[str, str]:
    parsed = _parse_grounded_answer_from_text(text)
    if parsed is not None:
        return parsed.answer, format_reasoning_bullets(parsed.reasoning)
    return _split_reasoning_from_text(text)


def _parse_grounded_answer_from_text(
    text: str,
) -> GroundedAnswerStructuredResponse | None:
    cleaned = text.strip()
    if not cleaned:
        return None

    payload = _parse_grounded_answer_json(cleaned)
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

    answer, reasoning = _split_reasoning_from_text(cleaned)
    if not answer.strip():
        return None
    fallback_payload: dict[str, Any] = {"answer": answer}
    if reasoning and reasoning.strip():
        fallback_payload["reasoning"] = [reasoning.strip()]
    try:
        return GroundedAnswerStructuredResponse.model_validate(fallback_payload)
    except ValidationError:
        return None


def _parse_grounded_answer_json(text: str) -> dict[str, Any] | None:
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


def _split_reasoning_from_text(text: str) -> tuple[str, str]:
    if not text:
        return "", ""

    thinking_match = re.search(
        r"<thinking>(.*?)</thinking>", text, re.IGNORECASE | re.DOTALL
    )
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)

    reasoning = thinking_match.group(1).strip() if thinking_match else ""

    if answer_match:
        answer = answer_match.group(1).strip()
    elif thinking_match:
        answer = re.sub(
            r"<thinking>.*?</thinking>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        answer = re.sub(r"</?answer>", "", answer, flags=re.IGNORECASE).strip()
    else:
        answer = text.strip()

    return answer, reasoning


def _build_source_grounded_fallback(citations: list[dict[str, Any]]) -> str:
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
