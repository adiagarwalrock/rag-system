"""Centralized OpenAI provider initialization and factories."""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core import Settings as LlamaSettings
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI, OpenAIResponses

from app.core.config import settings

logger = logging.getLogger(__name__)

_CONFIGURED_SIGNATURE: tuple[Any, ...] | None = None
DEFAULT_REASONING_EFFORT = "medium"
SUPPORTED_REASONING_EFFORTS = {"low", "medium", "high"}
_MESSAGE_ROLE_MAP: dict[str, MessageRole] = {
    "system": MessageRole.SYSTEM,
    "developer": MessageRole.DEVELOPER,
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    "tool": MessageRole.TOOL,
    "function": MessageRole.FUNCTION,
    "model": MessageRole.MODEL,
    "chatbot": MessageRole.CHATBOT,
}


def normalize_reasoning_effort(reasoning_effort: str | None) -> str:
    effort = (reasoning_effort or "").strip().lower()
    if effort in SUPPORTED_REASONING_EFFORTS:
        return effort
    return DEFAULT_REASONING_EFFORT


def get_llm(
    *,
    model: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    timeout_seconds: float | None = None,
):
    """Return an OpenAI-compatible LLM instance for the configured API mode."""
    llm_model = model or settings.LLM_MODEL
    resolved_key = api_key or settings.ai_api_key
    llm_class = OpenAIResponses if settings.OPENAI_USE_RESPONSES else OpenAI
    kwargs: dict[str, Any] = {"model": llm_model, "api_key": resolved_key}
    if llm_class is OpenAIResponses and reasoning_effort is not None:
        kwargs["reasoning_options"] = {
            "effort": normalize_reasoning_effort(reasoning_effort)
        }
    if timeout_seconds is not None:
        kwargs["timeout"] = float(timeout_seconds)
    return llm_class(**kwargs)


def get_embeddings(*, model: str | None = None, api_key: str | None = None):
    """Return an OpenAI embedding model instance with configured dimensions."""
    embedding_model = model or settings.EMBEDDING_MODEL
    resolved_key = api_key or settings.ai_api_key
    embedding_kwargs: dict[str, Any] = {}
    if settings.EMBEDDING_OUTPUT_DIMENSION is not None:
        embedding_kwargs["dimensions"] = settings.EMBEDDING_OUTPUT_DIMENSION

    return OpenAIEmbedding(
        model=embedding_model,
        api_key=resolved_key,
        **embedding_kwargs,
    )


def invoke_llm_chat(
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
    """Invoke a chat completion through the centralized LlamaIndex provider."""
    llm = get_llm(model=model, reasoning_effort=reasoning_effort, timeout_seconds=timeout_seconds)
    messages = _to_chat_messages(input_messages)
    runtime_kwargs = _build_chat_runtime_kwargs(
        max_output_tokens=max_output_tokens,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
        safety_identifier=safety_identifier,
        user_tag=user_tag,
        timeout_seconds=timeout_seconds,
    )
    return llm.chat(messages, **runtime_kwargs)


def extract_chat_response_text(response: Any) -> str:
    """Extract plain text content from a LlamaIndex chat response."""
    message = getattr(response, "message", None)
    if message is not None:
        chunks: list[str] = []
        for block in getattr(message, "blocks", None) or []:
            if isinstance(block, TextBlock):
                text = (block.text or "").strip()
                if text:
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks).strip()

        message_content = getattr(message, "content", None)
        if message_content:
            return str(message_content).strip()

    text = _extract_response_output_text(response)
    if text:
        return text

    raw_response = getattr(response, "raw", None)
    if raw_response is not None:
        text = _extract_response_output_text(raw_response)
        if text:
            return text

    return ""


def _extract_response_output_text(response: Any) -> str:
    output_text = _read_field(response, "output_text")
    if output_text:
        return str(output_text).strip()

    output_items = _read_field(response, "output") or []
    chunks: list[str] = []
    for item in output_items:
        if _read_field(item, "type") != "message":
            continue
        for content_item in _read_field(item, "content") or []:
            text = _read_field(content_item, "text")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def _read_field(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _to_chat_messages(input_messages: list[dict[str, Any]]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for message in input_messages:
        if not isinstance(message, dict):
            continue
        role = _resolve_message_role(message.get("role"))
        content = message.get("content")
        blocks = _content_to_blocks(content)
        if blocks is not None:
            messages.append(ChatMessage(role=role, blocks=blocks))
            continue
        messages.append(ChatMessage(role=role, content=str(content or "")))

    if not messages:
        raise ValueError("input_messages must include at least one message")
    return messages


def _content_to_blocks(content: Any) -> list[Any] | None:
    if not isinstance(content, list):
        return None

    blocks: list[Any] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()
        if item_type == "input_text":
            text = str(item.get("text") or "").strip()
            if text:
                blocks.append(TextBlock(text=text))
            continue
        if item_type == "input_image":
            image_url = str(item.get("image_url") or "").strip()
            if image_url:
                blocks.append(ImageBlock(url=image_url))

    return blocks or None


def _resolve_message_role(value: Any) -> MessageRole:
    role = str(value or "user").strip().lower()
    return _MESSAGE_ROLE_MAP.get(role, MessageRole.USER)


def _build_chat_runtime_kwargs(
    *,
    max_output_tokens: int | None,
    prompt_cache_key: str | None,
    prompt_cache_retention: str | None,
    safety_identifier: str | None,
    user_tag: str | None,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if timeout_seconds is not None:
        kwargs["timeout"] = float(timeout_seconds)
    if user_tag:
        kwargs["user"] = user_tag

    if settings.OPENAI_USE_RESPONSES:
        kwargs["truncation"] = "disabled"
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if prompt_cache_key:
            kwargs["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention:
            kwargs["prompt_cache_retention"] = prompt_cache_retention
        if safety_identifier:
            kwargs["safety_identifier"] = safety_identifier
        return kwargs

    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    return kwargs


def initialize_ai_provider(force: bool = False) -> None:
    """Initialize LlamaIndex global LLM/embedding settings from central config."""
    global _CONFIGURED_SIGNATURE

    api_key = settings.ai_api_key
    if settings.is_openai_api_key_placeholder:
        raise RuntimeError("AI_API_KEY is required and cannot be a placeholder.")

    signature = (
        api_key,
        settings.LLM_MODEL,
        settings.EMBEDDING_MODEL,
        settings.EMBEDDING_OUTPUT_DIMENSION,
        settings.OPENAI_USE_RESPONSES,
    )
    if (
        not force
        and _CONFIGURED_SIGNATURE == signature
        and LlamaSettings.llm is not None
        and LlamaSettings.embed_model is not None
    ):
        return

    LlamaSettings.llm = get_llm(api_key=api_key)
    LlamaSettings.embed_model = get_embeddings(api_key=api_key)
    _CONFIGURED_SIGNATURE = signature

    llm_api_mode = "responses" if settings.OPENAI_USE_RESPONSES else "chat_completions"
    logger.info(
        "Initialized OpenAI provider (llm=%s, embedding=%s, llm_api=%s)",
        settings.LLM_MODEL,
        settings.EMBEDDING_MODEL,
        llm_api_mode,
    )
