"""OpenAI LLM provider — Concrete Strategy + Factory Method.

build_llm() is the Factory Method producing a LlamaIndex OpenAI/OpenAIResponses object.
invoke() and stream() are the Strategy implementations covering the full OpenAI feature
set: Responses API, prompt caching, safety identifiers, reasoning effort/summary,
structured output, and streaming reasoning deltas.
"""
from __future__ import annotations

from collections.abc import Generator
from logging import DEBUG, getLogger
from typing import Any, TypeVar

import openai as _openai
from llama_index.llms.openai import OpenAI, OpenAIResponses
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
)
from pydantic import BaseModel

from app.core.message_manager import (
    _to_chat_messages,
    extract_chat_response_reasoning,
    extract_chat_response_text,
    normalize_reasoning_effort,
    normalize_reasoning_summary,
)
from app.core.config import settings
from app.core.models.llm.base import LLMProvider

_S = TypeVar("_S", bound=BaseModel)
logger = getLogger(__name__)

# Module-level caches shared across all provider instances.
# Keyed by (api_key, timeout) — raw openai.OpenAI clients are thread-safe and
# expensive to construct (SSL handshake + connection pool setup).
_raw_client_cache: dict[tuple[str, float | None], _openai.OpenAI] = {}

# Keyed by schema class — avoids re-computing and re-mutating the same JSON
# schema dict on every structured output call for the same Pydantic model.
_schema_cache: dict[type[BaseModel], dict[str, Any]] = {}


class OpenAILLMProvider(LLMProvider):
    PROVIDER_NAME = "openai"

    def __init__(self, model_id: str, api_key: str, **kwargs: Any) -> None:
        super().__init__(model_id, api_key, **kwargs)
        # Cache keyed by (reasoning_effort, reasoning_summary, timeout_seconds).
        # LlamaIndex LLM objects are stateless wrappers — safe to reuse across calls
        # with the same construction parameters.
        self._llm_cache: dict[tuple[str | None, str | None, float | None], Any] = {}

    # ── Factory Method ────────────────────────────────────────────────────────

    @classmethod
    def api_key_from_settings(cls, settings: Any) -> str:
        return settings.openai_api_key

    def build_llm(
        self,
        *,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        timeout_seconds: float | None = None,
        **_kwargs: Any,
    ) -> Any:
        return self._build_openai_llm(
            model=self.model_id,
            api_key=self.api_key or settings.openai_api_key,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            timeout_seconds=timeout_seconds,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_openai_llm(
        self,
        *,
        model: str,
        api_key: str,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        """Return a cached LlamaIndex LLM instance for the given parameters."""
        cache_key = (reasoning_effort, reasoning_summary, timeout_seconds)
        cached = self._llm_cache.get(cache_key)
        if cached is not None:
            return cached

        bare_model = model.removeprefix("openai/")
        llm_class = OpenAIResponses if settings.OPENAI_USE_RESPONSES else OpenAI
        kwargs: dict[str, Any] = {"model": bare_model, "api_key": api_key}
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

        llm = llm_class(**kwargs)
        self._llm_cache[cache_key] = llm
        return llm

    def _build_chat_runtime_kwargs(
        self,
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

    def _get_raw_client(self, *, timeout: float | None) -> _openai.OpenAI:
        """Return a cached raw openai.OpenAI client for structured output calls."""
        api_key = self.api_key or settings.openai_api_key
        cache_key = (api_key, timeout)
        client = _raw_client_cache.get(cache_key)
        if client is None:
            client = _openai.OpenAI(
                api_key=api_key,
                **({"timeout": timeout} if timeout is not None else {}),
            )
            _raw_client_cache[cache_key] = client
        return client

    def _invoke_structured(
        self,
        *,
        model: str,
        input_messages: list[dict[str, Any]],
        schema: type[_S],
        max_output_tokens: int | None,
        timeout_seconds: float | None,
    ) -> _S:
        """Call OpenAI structured output and return a parsed Pydantic instance."""
        timeout = float(timeout_seconds) if timeout_seconds is not None else None
        client = self._get_raw_client(timeout=timeout)

        if settings.OPENAI_USE_RESPONSES:
            json_schema = self._get_json_schema(schema)
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
            return schema.model_validate_json(response.output_text)

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

    @staticmethod
    def _read_field(value: Any, field_name: str) -> Any:
        if isinstance(value, dict):
            return value.get(field_name)
        return getattr(value, field_name, None)

    @staticmethod
    def _get_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
        """Return a cached, OpenAI-strict-mode JSON schema for the given Pydantic model."""
        cached = _schema_cache.get(schema)
        if cached is not None:
            return cached
        result = OpenAILLMProvider._add_additional_properties_false(schema.model_json_schema())
        _schema_cache[schema] = result
        return result

    @staticmethod
    def _add_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively add additionalProperties: false to all object schemas (OpenAI strict mode)."""
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)
        for sub in schema.get("properties", {}).values():
            if isinstance(sub, dict):
                OpenAILLMProvider._add_additional_properties_false(sub)
        for sub in schema.get("$defs", {}).values():
            if isinstance(sub, dict):
                OpenAILLMProvider._add_additional_properties_false(sub)
        return schema

    # ── Strategy ──────────────────────────────────────────────────────────────

    def _prepare(
        self,
        input_messages: list[dict[str, Any]],
        *,
        reasoning_effort: str | None,
        reasoning_summary: str | None,
        max_output_tokens: int | None,
        timeout_seconds: float | None,
        prompt_cache_key: str | None,
        prompt_cache_retention: str | None,
        safety_identifier: str | None,
        user_tag: str | None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Return (llm, messages, runtime_kwargs) ready for .chat() or .stream_chat()."""
        llm = self._build_openai_llm(
            model=self.model_id,
            api_key=self.api_key,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            timeout_seconds=timeout_seconds,
        )
        messages = _to_chat_messages(input_messages)
        runtime_kwargs = self._build_chat_runtime_kwargs(
            max_output_tokens=max_output_tokens,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            safety_identifier=safety_identifier,
            user_tag=user_tag,
            timeout_seconds=timeout_seconds,
        )
        return llm, messages, runtime_kwargs

    def invoke(
        self,
        input_messages: list[dict[str, Any]],
        *,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        max_output_tokens: int | None = None,
        timeout_seconds: float | None = None,
        structured_output_schema: type[_S] | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        safety_identifier: str | None = None,
        user_tag: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        if structured_output_schema is not None:
            return self._invoke_structured(
                model=self.model_id,
                input_messages=input_messages,
                schema=structured_output_schema,
                max_output_tokens=max_output_tokens,
                timeout_seconds=timeout_seconds,
            )

        llm, messages, runtime_kwargs = self._prepare(
            input_messages,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            safety_identifier=safety_identifier,
            user_tag=user_tag,
        )
        return llm.chat(messages, **runtime_kwargs)

    def stream(
        self,
        input_messages: list[dict[str, Any]],
        *,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        max_output_tokens: int | None = None,
        timeout_seconds: float | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        safety_identifier: str | None = None,
        user_tag: str | None = None,
        **_kwargs: Any,
    ) -> Generator[tuple[str | None, str | None], None, None]:
        llm, messages, runtime_kwargs = self._prepare(
            input_messages,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            safety_identifier=safety_identifier,
            user_tag=user_tag,
        )

        if not isinstance(llm, OpenAIResponses):
            response = llm.chat(messages, **runtime_kwargs)
            yield (None, extract_chat_response_text(response))
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

            if isinstance(
                raw_event,
                (ResponseReasoningSummaryTextDeltaEvent, ResponseReasoningTextDeltaEvent),
            ):
                reasoning_seen = True
                if raw_event.delta:
                    yield (raw_event.delta, None)
                continue

            raw_event_type = self._read_field(raw_event, "type")
            if raw_event_type in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            }:
                reasoning_seen = True
                delta = self._read_field(raw_event, "delta")
                if delta:
                    yield (str(delta), None)
                continue

            if isinstance(raw_event, ResponseReasoningSummaryTextDoneEvent):
                if not reasoning_seen and raw_event.text:
                    reasoning_seen = True
                    yield (raw_event.text, None)
                continue

            if raw_event_type == "response.reasoning_summary_text.done":
                text = self._read_field(raw_event, "text")
                if not reasoning_seen and text:
                    reasoning_seen = True
                    yield (str(text), None)
                continue

            if chunk.delta:
                yield (None, chunk.delta)
                continue

            if not reasoning_seen and isinstance(raw_event, ResponseCompletedEvent):
                reasoning_text = extract_chat_response_reasoning(chunk)
                if reasoning_text:
                    logger.debug(
                        "stream_chat: emitting ThinkingBlock content as reasoning delta (%d chars)",
                        len(reasoning_text),
                    )
                    reasoning_seen = True
                    yield (reasoning_text, None)

        logger.info("stream complete: reasoning_seen=%s", reasoning_seen)
