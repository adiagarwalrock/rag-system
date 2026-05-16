from __future__ import annotations

import logging
from typing import Any

from app.agents.agent_base import invoke_llm_chat
from app.core.config import settings
from app.prompts.registry import _build_grounded_prompt
from app.schemas.retrieval import GroundedAnswerResult, GroundedAnswerStructuredResponse
from app.services.synth_parse import (
    coerce_grounded_answer_structured_output,
    effective_max_output_tokens,
    extract_answer_and_reasoning_from_chat,
    to_grounded_answer_result,
)

logger = logging.getLogger(__name__)


def try_chat_synthesis(
    *,
    reasoning_effort: str,
    conversation_context: dict[str, Any],
    question: str,
    citations: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    image_paths: list[str],
    citation_image_map: dict[int, list[int]],
    build_grounded_message,
) -> GroundedAnswerResult | None:
    prompt = _build_grounded_prompt(
        question,
        citations,
        conflicts,
        image_attachment_count=len(image_paths),
        conversation_context=conversation_context,
        citation_image_map=citation_image_map,
    )

    attempts: list[tuple[str, Any, list[str]]] = []
    if image_paths:
        attempts.append(("multimodal", build_grounded_message(prompt, image_paths), image_paths))
    attempts.append(("text_only", build_grounded_message(prompt, []), []))

    for attempt_name, message, used_images in attempts:
        try:
            response = invoke_llm_chat(
                model=settings.LLM_MODEL,
                messages=[message],
                structured_output_cls=GroundedAnswerStructuredResponse,
                reasoning_effort=reasoning_effort,
                max_output_tokens=effective_max_output_tokens(question),
            )

            structured_payload = coerce_grounded_answer_structured_output(response)
            if structured_payload is not None:
                return to_grounded_answer_result(
                    structured_payload,
                    image_paths=used_images,
                    effort_applied=settings.OPENAI_USE_RESPONSES,
                    evidence_count=len(citations),
                )

            logger.warning(
                "%s structured synthesis yielded invalid payload; attempting manual extraction",
                attempt_name,
            )
            answer, reasoning = extract_answer_and_reasoning_from_chat(response)
            parsed_chat = GroundedAnswerStructuredResponse.model_validate(
                {
                    "answer": answer,
                    "reasoning": [reasoning] if reasoning else [],
                }
            )
            return to_grounded_answer_result(
                parsed_chat,
                image_paths=used_images,
                effort_applied=settings.OPENAI_USE_RESPONSES,
                evidence_count=len(citations),
            )
        except Exception:
            logger.exception("%s answer synthesis failed", attempt_name)

    return None
