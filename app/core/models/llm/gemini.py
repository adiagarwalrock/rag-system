"""Gemini LLM provider — Concrete Strategy + Factory Method.

Wraps llama-index-llms-google-genai (GoogleGenAI class).

Feature parity with OpenAI and Anthropic providers:
- max_output_tokens  → GoogleGenAI(max_tokens=...)
- structured output  → chat() with generation_config response_mime_type + response_schema
- reasoning          → thought blocks emitted as (reasoning_text, None) at stream end
- image messages     → ImageBlock in ChatMessage.blocks (handled by LlamaIndex wrapper)
- history            → multi-turn ChatMessage list passed directly to .chat()

Note: Gemini has no timeout constructor param. Thought blocks are not streamed as
deltas — they are emitted once at stream completion when present in the response.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any, TypeVar

from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel

from app.core.message_manager import _to_chat_messages, extract_chat_response_reasoning
from app.core.models.llm.base import LLMProvider

_S = TypeVar("_S", bound=BaseModel)


class GeminiLLMProvider(LLMProvider):
    PROVIDER_NAME = "gemini"

    @classmethod
    def api_key_from_settings(cls, settings: Any) -> str:
        return settings.gemini_api_key

    # ── Factory Method ────────────────────────────────────────────────────────

    def build_llm(
        self,
        *,
        max_output_tokens: int | None = None,
        **_kwargs: Any,
    ) -> Any:
        bare_model = self.model_id.removeprefix("gemini/").removeprefix("vertex_ai/")
        kwargs: dict[str, Any] = {"model": bare_model, "api_key": self.api_key}
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        return GoogleGenAI(**kwargs)

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
        llm = self.build_llm(max_output_tokens=max_output_tokens)
        messages = _to_chat_messages(input_messages)
        if structured_output_schema is not None:
            # Pass schema via generation_config so Gemini returns JSON matching the model
            return llm.chat(
                messages,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": structured_output_schema,
                },
            )
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
        llm = self.build_llm(max_output_tokens=max_output_tokens)
        messages = _to_chat_messages(input_messages)
        reasoning_emitted = False
        last_chunk: Any = None
        for chunk in llm.stream_chat(messages):
            last_chunk = chunk
            if chunk.delta:
                yield (None, chunk.delta)
        # Emit accumulated thought blocks once at stream end
        if last_chunk is not None and not reasoning_emitted:
            reasoning_text = extract_chat_response_reasoning(last_chunk)
            if reasoning_text:
                yield (reasoning_text, None)
