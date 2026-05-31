"""Centralized OpenAI provider initialization and factories."""

from __future__ import annotations

from collections.abc import Generator
from logging import Logger, getLogger, DEBUG
from typing import Any, TypeVar

from pydantic import BaseModel

_S = TypeVar("_S", bound=BaseModel)

from llama_index.core import Settings as LlamaSettings
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
    ThinkingBlock,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI, OpenAIResponses
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
)

from app.core.config import settings

logger: Logger = getLogger(__name__)

_CONFIGURED_SIGNATURE: tuple[Any, ...] | None = None
DEFAULT_REASONING_EFFORT = "medium"
SUPPORTED_REASONING_EFFORTS = {"low", "medium", "high"}
SUPPORTED_REASONING_SUMMARIES = {"auto", "concise", "detailed"}
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


def normalize_reasoning_summary(
    reasoning_summary: str | None,
    fallback: str | None = None,
) -> str | None:
    """Normalize and validate a reasoning-summary verbosity value.

    Returns the normalized value if it is in ``SUPPORTED_REASONING_SUMMARIES``,
    otherwise falls back to ``fallback`` (applying the same validation), and finally
    returns ``None`` when neither value is valid.
    """
    for candidate in (reasoning_summary, fallback):
        value = (candidate or "").strip().lower() or None
        if value and value in SUPPORTED_REASONING_SUMMARIES:
            return value
    return None


def get_llm(
    *,
    model: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    timeout_seconds: float | None = None,
):
    """Return an OpenAI-compatible LLM instance for the configured API mode.

    When using the Responses API with a reasoning model (gpt-5.x / o-series) and
    ``reasoning_effort`` is set, ``reasoning_summary`` controls whether OpenAI
    returns a human-readable summary of its chain-of-thought.  Supported values:
    ``"auto"``, ``"concise"``, ``"detailed"``.  Falls back to
    ``settings.REASONING_SUMMARY`` when not supplied by the caller; omit the key
    entirely when the resolved value is ``None`` or unrecognised.
    """
    llm_model = model or settings.LLM_MODEL
    resolved_key = api_key or settings.ai_api_key
    llm_class = OpenAIResponses if settings.OPENAI_USE_RESPONSES else OpenAI
    kwargs: dict[str, Any] = {"model": llm_model, "api_key": resolved_key}
    if llm_class is OpenAIResponses and reasoning_effort is not None:
        reasoning_opts: dict[str, Any] = {
            "effort": normalize_reasoning_effort(reasoning_effort)
        }
        resolved_summary = normalize_reasoning_summary(
            reasoning_summary, fallback=settings.REASONING_SUMMARY
        )
        if resolved_summary:
            reasoning_opts["summary"] = resolved_summary
        kwargs["reasoning_options"] = reasoning_opts
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
    reasoning_summary: str | None = None,
    max_output_tokens: int | None = None,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: str | None = None,
    safety_identifier: str | None = None,
    user_tag: str | None = None,
    timeout_seconds: float | None = None,
    structured_output_schema: type[_S] | None = None,
) -> Any:
    """Invoke a chat completion through the centralized LlamaIndex provider.

    When ``structured_output_schema`` is a Pydantic model class, the raw OpenAI
    client is used directly (bypassing LlamaIndex) and the return value is a
    parsed instance of that model rather than a LlamaIndex ChatResponse.
    """
    if structured_output_schema is not None:
        return _invoke_structured(
            model=model,
            input_messages=input_messages,
            schema=structured_output_schema,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )
    llm = get_llm(
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        timeout_seconds=timeout_seconds,
    )
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


def _add_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively add additionalProperties=false to all object nodes in a JSON schema.

    Required by OpenAI strict mode for the Responses API.
    """
    if schema.get("type") == "object":
        schema = {**schema, "additionalProperties": False}
    for key in ("properties", "definitions", "$defs"):
        if key in schema:
            schema = {**schema, key: {k: _add_additional_properties_false(v) for k, v in schema[key].items()}}
    for key in ("items", "anyOf", "allOf", "oneOf"):
        if key in schema:
            val = schema[key]
            if isinstance(val, list):
                schema = {**schema, key: [_add_additional_properties_false(v) if isinstance(v, dict) else v for v in val]}
            elif isinstance(val, dict):
                schema = {**schema, key: _add_additional_properties_false(val)}
    return schema


def _invoke_structured(
    *,
    model: str,
    input_messages: list[dict[str, Any]],
    schema: type[_S],
    max_output_tokens: int | None,
    timeout_seconds: float | None,
) -> _S:
    """Call OpenAI structured output and return a parsed Pydantic instance.

    Uses the Responses API when OPENAI_USE_RESPONSES=True, otherwise falls back
    to the Chat Completions beta parse endpoint.
    """
    import openai

    api_key = settings.ai_api_key
    timeout = float(timeout_seconds) if timeout_seconds is not None else None
    client = openai.OpenAI(api_key=api_key, **({"timeout": timeout} if timeout else {}))

    if settings.OPENAI_USE_RESPONSES:
        # Responses API: structured output via text.format = json_schema
        # OpenAI strict mode requires additionalProperties=false at every object level.
        json_schema = _add_additional_properties_false(schema.model_json_schema())
        kwargs: dict[str, Any] = {
            "model": model,
            "input": input_messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "schema": json_schema,
                    "strict": True,
                }
            },
        }
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        response = client.responses.create(**kwargs)
        text = response.output_text
        return schema.model_validate_json(text)

    # Chat Completions beta parse endpoint
    parse_kwargs: dict[str, Any] = {
        "model": model,
        "messages": input_messages,
        "response_format": schema,
    }
    if max_output_tokens is not None:
        parse_kwargs["max_tokens"] = max_output_tokens
    completion = client.beta.chat.completions.parse(**parse_kwargs)
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Structured output parse returned None")
    return parsed


def stream_invoke_llm_chat(
    *,
    model: str,
    input_messages: list[dict[str, Any]],
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    max_output_tokens: int | None = None,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: str | None = None,
    safety_identifier: str | None = None,
    user_tag: str | None = None,
    timeout_seconds: float | None = None,
) -> Generator[tuple[str | None, str | None], None, None]:
    """Stream a chat completion, yielding ``(reasoning_delta, answer_delta)`` tuples.

    Each tuple has at most one non-None field per event:
    - ``reasoning_delta`` — an incremental token of the reasoning *summary* (from
      ``ResponseReasoningSummaryTextDeltaEvent``).  Only fired when the model
      produces a reasoning summary (i.e. ``reasoning_summary`` is set and the model
      is a reasoning model such as gpt-5.x / o-series).
    - ``answer_delta`` — an incremental token of the final answer text.

    The generator is exhausted once ``ResponseCompletedEvent`` arrives.  Callers
    should accumulate both streams independently; the final answer and reasoning are
    available in full from the last ``ThinkingBlock`` / ``TextBlock`` on the
    ``ResponseCompletedEvent`` yield, but it is simpler to just accumulate deltas.

    Only works when ``OPENAI_USE_RESPONSES=True``; falls back to a single
    ``(None, full_answer)`` yield otherwise.
    """
    llm = get_llm(
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        timeout_seconds=timeout_seconds,
    )
    messages = _to_chat_messages(input_messages)
    runtime_kwargs = _build_chat_runtime_kwargs(
        max_output_tokens=max_output_tokens,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
        safety_identifier=safety_identifier,
        user_tag=user_tag,
        timeout_seconds=timeout_seconds,
    )

    if not isinstance(llm, OpenAIResponses):
        response = llm.chat(messages, **runtime_kwargs)
        full_text = extract_chat_response_text(response)
        yield (None, full_text)
        return

    reasoning_seen = False
    for chunk in llm.stream_chat(messages, **runtime_kwargs):
        raw_event = getattr(chunk, "raw", None)

        if logger.isEnabledFor(DEBUG) and isinstance(
            raw_event,
            (ResponseReasoningSummaryTextDeltaEvent, ResponseReasoningTextDeltaEvent),
        ):
            logger.debug(
                "stream_chat reasoning delta: raw_type=%s delta=%r",
                type(raw_event).__name__,
                raw_event.delta,
            )

        # Reasoning delta — emitted when reasoning.summary is set (summary event) or
        # as a fallback raw delta when summary mode is not active.
        if isinstance(
            raw_event,
            (ResponseReasoningSummaryTextDeltaEvent, ResponseReasoningTextDeltaEvent),
        ):
            reasoning_seen = True
            delta = raw_event.delta
            if delta:
                yield (delta, None)
            continue

        raw_event_type = _read_field(raw_event, "type")
        if raw_event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            reasoning_seen = True
            delta = _read_field(raw_event, "delta")
            if delta:
                yield (str(delta), None)
            continue

        # Some Responses streams deliver the full reasoning summary only in the
        # summary_text.done event. Emit it when no delta event already covered it.
        if isinstance(raw_event, ResponseReasoningSummaryTextDoneEvent):
            if not reasoning_seen and raw_event.text:
                reasoning_seen = True
                yield (raw_event.text, None)
            continue

        if raw_event_type == "response.reasoning_summary_text.done":
            text = _read_field(raw_event, "text")
            if not reasoning_seen and text:
                reasoning_seen = True
                yield (str(text), None)
            continue

        # Incremental answer token.
        if chunk.delta:
            yield (None, chunk.delta)
            continue

        # On the final ResponseCompletedEvent, LlamaIndex replaces blocks with the full
        # parsed output (including ThinkingBlocks from reasoning items). If no reasoning
        # deltas were received yet (e.g. model uses a different event format), emit the
        # ThinkingBlock content now so callers see reasoning.
        if not reasoning_seen and isinstance(raw_event, ResponseCompletedEvent):
            reasoning_text = extract_chat_response_reasoning(chunk)
            if reasoning_text:
                logger.debug(
                    "stream_chat: emitting ThinkingBlock content as reasoning delta (%d chars)",
                    len(reasoning_text),
                )
                reasoning_seen = True
                yield (reasoning_text, None)

    logger.info(
        "stream_invoke_llm_chat complete: reasoning_seen=%s",
        reasoning_seen,
    )


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
    """Extract reasoning summary text from a LlamaIndex chat response.

    When ``REASONING_SUMMARY`` is set, the Responses API returns a
    ``ResponseReasoningItem`` that LlamaIndex converts into a ``ThinkingBlock``
    in ``response.message.blocks``.  This function extracts its ``content`` field,
    which contains the concatenated reasoning summary text.

    Returns ``None`` when no reasoning block is present.
    """
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
