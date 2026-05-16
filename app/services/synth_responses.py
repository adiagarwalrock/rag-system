from __future__ import annotations

import logging
from typing import Any

from app.agents.agent_base import extract_chat_response_text, invoke_llm_chat
from app.core.config import settings
from app.prompts.registry import _build_labeled_context_sections
from app.prompts.templates import GROUNDED_ANSWER_DEVELOPER_PROMPT
from app.schemas.retrieval import GroundedAnswerResult, GroundedAnswerStructuredResponse
from observability.cost_tracker import ResponsesInputBudgeter
from app.services.synth_parse import (
    coerce_grounded_answer_structured_output,
    effective_max_output_tokens,
    is_incomplete_structured_output_error,
    parse_grounded_answer_from_text,
    to_grounded_answer_result,
)

logger = logging.getLogger(__name__)


def try_responses_synthesis(
    *,
    client_id: str,
    reasoning_effort: str,
    conversation_context: dict[str, Any],
    question: str,
    citations: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    image_paths: list[str],
    append_image_inputs,
) -> GroundedAnswerResult | None:
    try:
        budgeter = ResponsesInputBudgeter(model=settings.LLM_MODEL)
        sections = _build_labeled_context_sections(
            citations=citations,
            conflicts=conflicts,
            conversation_context=conversation_context,
        )
        input_messages, _, metrics = budgeter.build_budgeted_sections(
            developer_prompt=GROUNDED_ANSWER_DEVELOPER_PROMPT,
            question=question,
            recent_turns=conversation_context.get("recent_turns") or [],
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
            input_messages = append_image_inputs(
                input_messages=input_messages,
                image_paths=image_paths,
            )

        structured_response = None
        try:
            structured_response = invoke_llm_chat(
                model=settings.LLM_MODEL,
                input_messages=input_messages,
                reasoning_effort=reasoning_effort,
                max_output_tokens=effective_max_output_tokens(question),
                prompt_cache_key=settings.RESPONSE_PROMPT_CACHE_KEY,
                prompt_cache_retention=settings.RESPONSE_PROMPT_CACHE_RETENTION,
                safety_identifier=f"{settings.RESPONSE_SAFETY_IDENTIFIER_PREFIX}:{client_id}",
                user_tag=settings.RESPONSE_USER_TAG,
                structured_output_cls=GroundedAnswerStructuredResponse,
            )
        except Exception as exc:
            if is_incomplete_structured_output_error(exc):
                logger.warning(
                    "Structured grounded answer output incomplete; retrying with parser fallback"
                )
            else:
                logger.exception(
                    "Structured grounded answer synthesis failed; retrying with parser fallback"
                )

        if structured_response is not None:
            structured_payload = coerce_grounded_answer_structured_output(
                structured_response
            )
            if structured_payload is not None:
                return to_grounded_answer_result(
                    structured_payload,
                    image_paths=image_paths,
                    effort_applied=True,
                    evidence_count=len(citations),
                )

            logger.warning(
                "Structured grounded answer payload invalid; using parser fallback on response text"
            )
            structured_fallback = coerce_grounded_answer_structured_output(
                extract_chat_response_text(structured_response)
            )
            if structured_fallback is not None:
                return to_grounded_answer_result(
                    structured_fallback,
                    image_paths=image_paths,
                    effort_applied=True,
                    evidence_count=len(citations),
                )

        fallback_response = invoke_llm_chat(
            model=settings.LLM_MODEL,
            input_messages=input_messages,
            reasoning_effort=reasoning_effort,
            max_output_tokens=effective_max_output_tokens(question),
            prompt_cache_key=settings.RESPONSE_PROMPT_CACHE_KEY,
            prompt_cache_retention=settings.RESPONSE_PROMPT_CACHE_RETENTION,
            safety_identifier=f"{settings.RESPONSE_SAFETY_IDENTIFIER_PREFIX}:{client_id}",
            user_tag=settings.RESPONSE_USER_TAG,
        )
        parsed_fallback = parse_grounded_answer_from_text(
            extract_chat_response_text(fallback_response)
        )
        if parsed_fallback is None:
            raise ValueError("LLM returned empty answer")
        return to_grounded_answer_result(
            parsed_fallback,
            image_paths=image_paths,
            effort_applied=True,
            evidence_count=len(citations),
        )
    except Exception:
        logger.exception("Responses answer synthesis failed; falling back to LlamaIndex chat")
        return None
