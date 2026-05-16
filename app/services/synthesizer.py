"""
Grounded answer synthesis via LLM.

Orchestrates synthesis paths while delegating parsing and provider-specific
logic to dedicated modules.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole, TextBlock

from app.core.config import settings
from app.prompts.registry import _collect_image_evidence_paths
from app.schemas.retrieval import GroundedAnswerResult
from app.services.synth_chat import try_chat_synthesis
from app.services.synth_parse import (
    build_source_grounded_fallback,
    effective_max_output_tokens as _effective_max_output_tokens,
    extract_answer_and_reasoning_from_chat as _extract_answer_and_reasoning_from_chat,
    normalize_inline_citations as _normalize_inline_citations,
    scrub_cot_artifacts as _scrub_cot_artifacts,
    split_reasoning_from_text as _split_reasoning_from_text_answer_first,
)
from app.services.synth_responses import try_responses_synthesis

logger = logging.getLogger(__name__)


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
            responses_result = try_responses_synthesis(
                client_id=self.client_id,
                reasoning_effort=self.reasoning_effort,
                conversation_context=self.conversation_context,
                question=question,
                citations=citations,
                conflicts=conflicts,
                image_paths=image_paths,
                append_image_inputs=_append_image_inputs,
            )
            if responses_result is not None:
                return responses_result.to_dict()

        chat_result = try_chat_synthesis(
            reasoning_effort=self.reasoning_effort,
            conversation_context=self.conversation_context,
            question=question,
            citations=citations,
            conflicts=conflicts,
            image_paths=image_paths,
            citation_image_map=citation_image_map,
            build_grounded_message=_build_grounded_message,
        )
        if chat_result is not None:
            return chat_result.to_dict()

        logger.error("Answer synthesis failed; returning source-grounded fallback")
        return GroundedAnswerResult(
            answer=build_source_grounded_fallback(citations),
            reasoning="",
            images_used=image_paths,
            reasoning_effort_applied=effort_applied,
        ).to_dict()


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
        from app.prompts.registry import _image_path_to_data_url

        data_url = _image_path_to_data_url(path)
        if not data_url:
            continue
        multimodal_content.append({"type": "input_image", "image_url": data_url})

    user_message["content"] = multimodal_content
    return messages


# Backward-compatible names used by tests / internal imports
# These are symbol aliases to the new split modules, not implementation wrappers.

_normalize_inline_citations = _normalize_inline_citations
_extract_answer_and_reasoning_from_chat = _extract_answer_and_reasoning_from_chat
_effective_max_output_tokens = _effective_max_output_tokens
_scrub_cot_artifacts = _scrub_cot_artifacts


def _split_reasoning_from_text(text: str) -> tuple[str, str]:
    """Legacy test-facing order: (reasoning, answer)."""
    answer, reasoning = _split_reasoning_from_text_answer_first(text)
    return reasoning, answer
