from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any, TypeVar

from llama_index.core import Settings as LlamaSettings
from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole, TextBlock
from pydantic import BaseModel

from app.core.config import settings
from app.core.structured_output import coerce_structured_output
from app.prompts.templates import _GENERIC_STRUCTURED_PROMPT

logger = logging.getLogger(__name__)

_Model = TypeVar("_Model", bound=BaseModel)
_REASONING_RETRY_TOKEN_CAP = 1200
_REASONING_RETRY_TOKEN_INCREMENT = 350


def is_transient_provider_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    error_text = str(exc).lower()
    return status_code in {429, 500, 502, 503, 504} or any(
        marker in error_text
        for marker in (
            "deadline expired",
            "timed out",
            "timeout",
            "unavailable",
            "temporarily unavailable",
            "try again",
        )
    )


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


def reasoning_retry_output_tokens(max_output_tokens: int | None) -> int | None:
    if max_output_tokens is None or max_output_tokens <= 0:
        return None
    scaled = int(max_output_tokens * 1.6)
    expanded = max(max_output_tokens + _REASONING_RETRY_TOKEN_INCREMENT, scaled)
    return min(_REASONING_RETRY_TOKEN_CAP, max(max_output_tokens, expanded))


def coerce_output(output: Any, output_cls: type[_Model]) -> _Model | None:
    return coerce_structured_output(output, output_cls)


def run_structured_text_inference(prompt: str, output_cls: type[_Model]) -> _Model | None:
    try:
        structured = LlamaSettings.llm.structured_predict(
            output_cls,
            _GENERIC_STRUCTURED_PROMPT,
            user_prompt=prompt,
        )
        return coerce_output(structured, output_cls)
    except Exception:
        logger.exception("Structured text inference failed for %s", output_cls.__name__)
        return None


def run_structured_multimodal_inference(
    *,
    prompt: str,
    image_path: str,
    output_cls: type[_Model],
) -> _Model | None:
    if (
        not image_path
        or settings.is_openai_api_key_placeholder
        or not Path(image_path).exists()
    ):
        return None

    try:
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        structured_llm = LlamaSettings.llm.as_structured_llm(output_cls)
        response = structured_llm.chat(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    blocks=[
                        TextBlock(text=prompt),
                        ImageBlock(path=Path(image_path), image_mimetype=mime_type),
                    ],
                )
            ]
        )
        parsed = coerce_output(response, output_cls)
        if parsed is not None:
            return parsed
    except Exception as exc:
        if is_transient_provider_error(exc):
            logger.warning(
                "Structured multimodal inference unavailable for %s (status=%s): %s",
                image_path,
                getattr(exc, "status_code", None),
                exc,
            )
        else:
            logger.exception(
                "Structured multimodal inference failed for %s", image_path
            )

    return None
