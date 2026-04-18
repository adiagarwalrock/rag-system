from __future__ import annotations

from typing import Any

from openai import OpenAI

from app.core.ai_provider import normalize_reasoning_effort
from app.core.config import settings

_CLIENT: OpenAI | None = None


def get_openai_client(api_key: str | None = None) -> OpenAI:
    global _CLIENT
    resolved_key = api_key or settings.ai_api_key
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=resolved_key)
    return _CLIENT


def create_responses_completion(
    *,
    model: str,
    input_messages: list[dict[str, Any]],
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: str | None = None,
    safety_identifier: str | None = None,
    user_tag: str | None = None,
    timeout_seconds: float | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": input_messages,
        "max_output_tokens": max_output_tokens or settings.RESPONSE_MAX_OUTPUT_TOKENS,
        "truncation": "disabled",
    }
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": normalize_reasoning_effort(reasoning_effort)}
    if prompt_cache_key:
        kwargs["prompt_cache_key"] = prompt_cache_key
    if prompt_cache_retention:
        kwargs["prompt_cache_retention"] = prompt_cache_retention
    if safety_identifier:
        kwargs["safety_identifier"] = safety_identifier
    if user_tag:
        kwargs["user"] = user_tag

    client = get_openai_client()
    return client.responses.create(
        **kwargs,
        timeout=timeout_seconds,
    )


def extract_response_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    output_items = getattr(response, "output", None) or []
    chunks: list[str] = []
    for item in output_items:
        if getattr(item, "type", None) != "message":
            continue
        for content_item in getattr(item, "content", None) or []:
            text = getattr(content_item, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()
