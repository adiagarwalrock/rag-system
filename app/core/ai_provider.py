"""Centralized AI provider initialization and factory delegation."""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core import Settings as LlamaSettings
from llama_index.core.base.llms.types import (
    TextBlock,
)

from app.core.ai_factory import DEFAULT_REASONING_EFFORT as _DEFAULT_REASONING_EFFORT
from app.core.ai_factory import GEMINI_PROVIDER
from app.core.ai_factory import (
    SUPPORTED_REASONING_EFFORTS as _SUPPORTED_REASONING_EFFORTS,
)
from app.core.ai_factory import (
    AIProviderFactoryResolver,
    is_placeholder_api_key,
    normalize_reasoning_effort,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

_CONFIGURED_SIGNATURE: tuple[Any, ...] | None = None
DEFAULT_REASONING_EFFORT = _DEFAULT_REASONING_EFFORT
SUPPORTED_REASONING_EFFORTS = _SUPPORTED_REASONING_EFFORTS


def _build_provider_resolver() -> AIProviderFactoryResolver:
    return AIProviderFactoryResolver(
        openai_api_key=getattr(settings, "openai_api_key", ""),
        gemini_api_key=getattr(settings, "gemini_api_key", ""),
        google_api_key=getattr(settings, "google_api_key", ""),
    )


def _resolve_factory(model: str | None) -> tuple[str, Any, AIProviderFactoryResolver]:
    resolver = _build_provider_resolver()
    provider = resolver.resolve_provider(model=model)
    factory = resolver.get_factory(provider)
    return provider, factory, resolver


def get_llm(
    *,
    model: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
):
    """Return an LLM instance for the resolved provider."""
    llm_model = model or settings.LLM_MODEL
    provider, factory, resolver = _resolve_factory(llm_model)
    resolved_key = resolver.resolve_api_key(
        provider=provider,
        fallback_key=api_key or settings.ai_api_key,
    )
    return factory.create_llm(
        model=llm_model,
        api_key=resolved_key,
        reasoning_effort=reasoning_effort,
        use_responses_api=settings.OPENAI_USE_RESPONSES,
    )


def get_embeddings(*, model: str | None = None, api_key: str | None = None):
    """Return an embedding instance for the resolved provider."""
    embedding_model = model or settings.EMBEDDING_MODEL
    provider, factory, resolver = _resolve_factory(embedding_model)
    resolved_key = resolver.resolve_api_key(
        provider=provider,
        fallback_key=api_key or settings.ai_api_key,
    )
    return factory.create_embedding(
        model=embedding_model,
        api_key=resolved_key,
        output_dimension=settings.EMBEDDING_OUTPUT_DIMENSION,
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
    provider, factory, _ = _resolve_factory(model)
    llm = get_llm(model=model, reasoning_effort=reasoning_effort)
    messages = factory.to_chat_messages(input_messages)
    runtime_kwargs = factory.build_chat_runtime_kwargs(
        max_output_tokens=max_output_tokens,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
        safety_identifier=safety_identifier,
        user_tag=user_tag,
        timeout_seconds=timeout_seconds,
        use_responses_api=settings.OPENAI_USE_RESPONSES,
    )

    if provider == GEMINI_PROVIDER:
        runtime_kwargs.pop("timeout", None)
        runtime_kwargs.pop("user", None)

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


def initialize_ai_provider(force: bool = False) -> None:
    """Initialize LlamaIndex global LLM/embedding settings from central config."""
    global _CONFIGURED_SIGNATURE

    llm_provider, _, resolver = _resolve_factory(settings.LLM_MODEL)
    embedding_provider, _, _ = _resolve_factory(settings.EMBEDDING_MODEL)

    llm_api_key = resolver.resolve_api_key(
        provider=llm_provider,
        fallback_key=settings.ai_api_key,
    )
    embedding_api_key = resolver.resolve_api_key(
        provider=embedding_provider,
        fallback_key=settings.ai_api_key,
    )

    if is_placeholder_api_key(llm_api_key):
        raise RuntimeError("LLM API key is required and cannot be a placeholder.")
    if is_placeholder_api_key(embedding_api_key):
        raise RuntimeError("Embedding API key is required and cannot be a placeholder.")

    signature = (
        llm_provider,
        embedding_provider,
        llm_api_key,
        embedding_api_key,
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

    LlamaSettings.llm = get_llm(api_key=llm_api_key)
    LlamaSettings.embed_model = get_embeddings(api_key=embedding_api_key)
    _CONFIGURED_SIGNATURE = signature

    if llm_provider == GEMINI_PROVIDER:
        llm_api_mode = "google_genai"
    else:
        llm_api_mode = (
            "responses" if settings.OPENAI_USE_RESPONSES else "chat_completions"
        )

    logger.info(
        "Initialized AI providers (llm_provider=%s llm=%s embedding_provider=%s embedding=%s llm_api=%s)",
        llm_provider,
        settings.LLM_MODEL,
        embedding_provider,
        settings.EMBEDDING_MODEL,
        llm_api_mode,
    )
