"""Anthropic LLM provider — Concrete Strategy + Factory Method.

Wraps llama-index-llms-anthropic (Anthropic class).

Feature parity with OpenAI and Gemini providers:
- max_output_tokens  → Anthropic(max_tokens=...)
- timeout            → Anthropic(timeout=...)
- reasoning          → Anthropic(thinking_dict={...}); ThinkingBlock deltas in stream
- structured output  → llm.as_structured_llm(schema).chat(messages)
- image messages     → ImageBlock in ChatMessage.blocks (handled by LlamaIndex wrapper)
- history            → multi-turn ChatMessage list passed directly to .chat()
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any, TypeVar

from llama_index.llms.anthropic import Anthropic as _LlamaAnthropic
from pydantic import BaseModel

from app.core.message_manager import _to_chat_messages, normalize_reasoning_effort
from app.core.models.llm.base import LLMProvider

_S = TypeVar("_S", bound=BaseModel)

_THINKING_BUDGET: dict[str, int] = {"low": 1024, "medium": 4096, "high": 10000}
_DEFAULT_MAX_TOKENS = 4096


class AnthropicLLMProvider(LLMProvider):
    PROVIDER_NAME = "anthropic"

    @classmethod
    def api_key_from_settings(cls, settings: Any) -> str:
        return settings.anthropic_api_key

    # ── Factory Method ────────────────────────────────────────────────────────

    def build_llm(
        self,
        *,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        timeout_seconds: float | None = None,
        **_kwargs: Any,
    ) -> Any:
        bare = self.model_id.removeprefix("anthropic/")
        kwargs: dict[str, Any] = {
            "model": bare,
            "api_key": self.api_key,
            "max_tokens": max_output_tokens or _DEFAULT_MAX_TOKENS,
        }
        if timeout_seconds is not None:
            kwargs["timeout"] = float(timeout_seconds)
        if reasoning_effort is not None:
            budget = _THINKING_BUDGET.get(normalize_reasoning_effort(reasoning_effort), 4096)
            kwargs["thinking_dict"] = {"type": "enabled", "budget_tokens": budget}
        return _LlamaAnthropic(**kwargs)

    # ── Strategy ──────────────────────────────────────────────────────────────

    def invoke(
        self,
        input_messages: list[dict[str, Any]],
        *,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        max_output_tokens: int | None = None,
        timeout_seconds: float | None = None,
        structured_output_schema: type[_S] | None = None,
        **_kwargs: Any,
    ) -> Any:
        llm = self.build_llm(
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )
        messages = _to_chat_messages(input_messages)
        if structured_output_schema is not None:
            return llm.as_structured_llm(structured_output_schema).chat(messages)
        return llm.chat(messages)

    def stream(
        self,
        input_messages: list[dict[str, Any]],
        *,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        max_output_tokens: int | None = None,
        timeout_seconds: float | None = None,
        **_kwargs: Any,
    ) -> Generator[tuple[str | None, str | None], None, None]:
        llm = self.build_llm(
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )
        messages = _to_chat_messages(input_messages)
        for chunk in llm.stream_chat(messages):
            # ThinkingBlock deltas surface in additional_kwargs['thinking_delta']
            thinking_delta = (chunk.additional_kwargs or {}).get("thinking_delta")
            if thinking_delta:
                yield (thinking_delta, None)
            elif chunk.delta:
                yield (None, chunk.delta)
