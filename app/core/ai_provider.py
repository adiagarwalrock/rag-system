"""Centralized OpenAI provider initialization and factories."""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI, OpenAIResponses

from app.core.config import settings

logger = logging.getLogger(__name__)

_CONFIGURED_SIGNATURE: tuple[Any, ...] | None = None


def get_llm(*, model: str | None = None, api_key: str | None = None):
    """Return an OpenAI-compatible LLM instance for the configured API mode."""
    llm_model = model or settings.LLM_MODEL
    resolved_key = api_key or settings.ai_api_key
    llm_class = OpenAIResponses if settings.OPENAI_USE_RESPONSES else OpenAI
    return llm_class(model=llm_model, api_key=resolved_key)


def get_embedding_model(*, model: str | None = None, api_key: str | None = None):
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
    LlamaSettings.embed_model = get_embedding_model(api_key=api_key)
    _CONFIGURED_SIGNATURE = signature

    llm_api_mode = "responses" if settings.OPENAI_USE_RESPONSES else "chat_completions"
    logger.info(
        "Initialized OpenAI provider (llm=%s, embedding=%s, llm_api=%s)",
        settings.LLM_MODEL,
        settings.EMBEDDING_MODEL,
        llm_api_mode,
    )
