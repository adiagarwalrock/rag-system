"""Message utilities and shared LLM helpers used across providers.

This module has no upward dependencies on ai_provider or llm_manager, making it
safe to import from provider files without creating circular imports.

Exports:
- _to_chat_messages / _content_to_blocks / _resolve_message_role: dict → ChatMessage
- normalize_reasoning_effort / normalize_reasoning_summary: effort/summary validation
- extract_chat_response_text / extract_chat_response_reasoning: response parsing
"""
from __future__ import annotations

from typing import Any

from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
    ThinkingBlock,
)

# ── Reasoning normalization ────────────────────────────────────────────────────

DEFAULT_REASONING_EFFORT = "medium"
SUPPORTED_REASONING_EFFORTS = {"low", "medium", "high"}
SUPPORTED_REASONING_SUMMARIES = {"auto", "concise", "detailed"}


def normalize_reasoning_effort(reasoning_effort: str | None) -> str:
    effort = (reasoning_effort or "").strip().lower()
    if effort in SUPPORTED_REASONING_EFFORTS:
        return effort
    return DEFAULT_REASONING_EFFORT


def normalize_reasoning_summary(
    reasoning_summary: str | None,
    fallback: str | None = None,
) -> str | None:
    for candidate in (reasoning_summary, fallback):
        value = (candidate or "").strip().lower() or None
        if value and value in SUPPORTED_REASONING_SUMMARIES:
            return value
    return None


# ── Response extraction ────────────────────────────────────────────────────────

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


def extract_chat_response_reasoning(response: Any) -> str | None:
    """Extract reasoning summary text from a LlamaIndex chat response."""
    message = getattr(response, "message", None)
    if message is None:
        return None
    parts: list[str] = []
    for block in getattr(message, "blocks", None) or []:
        if isinstance(block, ThinkingBlock):
            text = (block.content or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip() or None


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


def _to_chat_messages(input_messages: list[dict[str, Any]]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for message in input_messages:
        if not isinstance(message, dict):
            continue
        role = _resolve_message_role(message.get("role"))
        content = message.get("content")

        # Preserve the OpenAI Responses API `phase` field on assistant messages.
        # Values: "commentary" (intermediate preamble before tool calls) or
        # "final_answer".  Dropping this on replayed assistant turns can cause
        # early-stopping with gpt-5.x models.
        additional_kwargs: dict[str, Any] = {}
        phase = message.get("phase")
        if phase and role == MessageRole.ASSISTANT:
            additional_kwargs["phase"] = phase

        blocks = _content_to_blocks(content)
        if blocks is not None:
            messages.append(
                ChatMessage(
                    role=role, blocks=blocks, additional_kwargs=additional_kwargs
                )
            )
            continue
        messages.append(
            ChatMessage(
                role=role,
                content=str(content or ""),
                additional_kwargs=additional_kwargs,
            )
        )

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
