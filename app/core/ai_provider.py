"""Shared LLM utilities, public factory functions, and thin call API.

Public entry points: invoke_llm_chat, stream_invoke_llm_chat, get_llm, get_embeddings,
initialize_ai_provider.

All provider-specific logic lives in app/core/models/llm/{openai,anthropic,gemini}.py.
This file contains only shared helpers used across providers and callers.
"""

from __future__ import annotations

from collections.abc import Generator
from logging import Logger, getLogger
from typing import Any, TypeVar

from pydantic import BaseModel

_S = TypeVar("_S", bound=BaseModel)

from llama_index.core import Settings as LlamaSettings
from app.core.config import settings
from app.core.embedding_manager import embedding_manager
from app.core.models.llm_manager import llm_manager
from app.core.models.embedding.base import EmbeddingProvider
from app.core.models.llm.base import LLMProvider as _LLMProvider

logger: Logger = getLogger(__name__)

_CONFIGURED_SIGNATURE: tuple[Any, ...] | None = None


def _is_openai_model(model: str) -> bool:
    """Return True for bare OpenAI names or 'openai/...' prefixed strings."""
    return model.startswith("openai/") or "/" not in model


def get_llm(
    *,
    model: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    timeout_seconds: float | None = None,
):
    """Return a LlamaIndex-compatible LLM instance.

    All providers (openai/*, anthropic/*, gemini/*) are routed through LLMManager.
    When reasoning_effort or timeout_seconds are set, the provider's build_llm() is
    called directly so those params are forwarded; otherwise the cached instance is
    returned from LLMManager.get_instance().

    For reasoning models with ``reasoning_effort`` set, ``reasoning_summary``
    controls chain-of-thought verbosity (``"auto"``, ``"concise"``, ``"detailed"``).
    """

    llm_model = model or settings.LLM_MODEL
    return llm_manager.get_instance(
        model_id=llm_model,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        timeout_seconds=timeout_seconds,
    )


def get_embeddings(*, model: str | None = None, api_key: str | None = None):
    """Return a LlamaIndex BaseEmbedding instance for the given model.

    Delegates to EmbeddingManager which selects the correct provider, resolves
    the API key from settings, and caches the instance for reuse.
    """
    return embedding_manager.get_instance(
        model_id=model or settings.EMBEDDING_MODEL,
        api_key=api_key,
        dimensions=settings.EMBEDDING_OUTPUT_DIMENSION,
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
    """Invoke a chat completion.

    Delegates to LLMManager (Facade) which resolves the correct provider strategy.
    When ``structured_output_schema`` is a Pydantic model class, the return value
    is a parsed instance of that model rather than a LlamaIndex ChatResponse.
    """

    return llm_manager.invoke(
        model_id=model,
        input_messages=input_messages,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        max_output_tokens=max_output_tokens,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
        safety_identifier=safety_identifier,
        user_tag=user_tag,
        timeout_seconds=timeout_seconds,
        structured_output_schema=structured_output_schema,
    )


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

    Only works when ``OPENAI_USE_RESPONSES=True`` for OpenAI models; falls back to a
    single ``(None, full_answer)`` yield otherwise.

    For non-OpenAI models (anthropic/*, gemini/*), yields answer deltas only.
    """

    yield from llm_manager.stream(
        model_id=model,
        input_messages=input_messages,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        max_output_tokens=max_output_tokens,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
        safety_identifier=safety_identifier,
        user_tag=user_tag,
        timeout_seconds=timeout_seconds,
    )


def initialize_ai_provider(force: bool = False) -> None:
    """Initialize LlamaIndex global LLM/embedding settings from central config."""
    global _CONFIGURED_SIGNATURE

    api_key = settings.openai_api_key
    if settings.is_openai_api_key_placeholder:
        raise RuntimeError("OPENAI_API_KEY is required and cannot be a placeholder.")

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

    embedding_provider = EmbeddingProvider.detect_provider(settings.EMBEDDING_MODEL)
    llm_provider = _LLMProvider.detect_provider(settings.LLM_MODEL)
    llm_api_mode = (
        "responses"
        if (settings.OPENAI_USE_RESPONSES and _is_openai_model(settings.LLM_MODEL))
        else "chat_completions"
    )
    logger.info(
        "Initialized provider (llm=%s [%s/%s], embedding=%s [%s])",
        settings.LLM_MODEL,
        llm_provider,
        llm_api_mode,
        settings.EMBEDDING_MODEL,
        embedding_provider,
    )
