"""LLM Manager — Facade + Singleton.

Acts as the single entry point for all LLM operations:
- invoke() / stream(): resolve the correct Concrete Strategy and delegate
- get_instance(): Factory Method path for LlamaIndex LLM objects (get_llm / global settings)
- list_models(): registry query
"""

from __future__ import annotations

from collections.abc import Generator
from threading import Lock
from typing import Any

from app.core.config import settings
from app.core.models.llm.base import (
    LLM_REGISTRY,
    LLMModelEntry,
    LLMProvider,
)
from app.core.models.llm.anthropic import AnthropicLLMProvider
from app.core.models.llm.gemini import GeminiLLMProvider
from app.core.models.llm.openai import OpenAILLMProvider

__all__ = [
    "LLMManager",
    "llm_manager",
    "get_llm",
    "AnthropicLLMProvider",
    "GeminiLLMProvider",
    "OpenAILLMProvider",
]

_PROVIDER_SORT_ORDER: dict[str, int] = {"openai": 0, "anthropic": 1, "gemini": 2}
_SORTED_LLM_REGISTRY: list[LLMModelEntry] = sorted(
    LLM_REGISTRY, key=lambda e: _PROVIDER_SORT_ORDER.get(e.provider, 3)
)


class LLMManager:
    """Factory and thread-safe cache for LlamaIndex-compatible LLM instances.

    Keyed by (model_id, api_key) — the same combination always returns the same
    object, avoiding redundant initialisation across concurrent requests.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str | None], Any] = {}
        self._lock = Lock()

    @staticmethod
    def _resolve_cls_and_key(
        model_id: str, api_key: str | None
    ) -> tuple[type[LLMProvider], str]:
        """Return (provider_cls, resolved_api_key) for model_id."""
        provider_cls = LLMProvider.for_provider(LLMProvider.detect_provider(model_id))
        return provider_cls, api_key or provider_cls.api_key_from_settings(settings)

    def get_instance(
        self,
        *,
        model_id: str,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        """Return a cached or freshly built LLM instance.

        Reasoning/timeout params are part of the cache key so callers with
        different reasoning options get distinct cached instances.
        """

        provider_cls, resolved_key = self._resolve_cls_and_key(model_id, api_key)
        cache_key = (
            model_id,
            resolved_key,
            reasoning_effort,
            reasoning_summary,
            timeout_seconds,
        )
        with self._lock:
            if cache_key not in self._cache:
                self._cache[cache_key] = provider_cls(
                    model_id=model_id, api_key=resolved_key
                ).build_llm(
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                    timeout_seconds=timeout_seconds,
                )
            return self._cache[cache_key]

    def invalidate(self, model_id: str | None = None) -> None:
        """Evict one model's cached instance, or all when model_id is None."""
        with self._lock:
            if model_id is None:
                self._cache.clear()
            else:
                self._cache = {k: v for k, v in self._cache.items() if k[0] != model_id}

    def list_models(self) -> list[LLMModelEntry]:
        """Return registry entries sorted openai → anthropic → gemini."""
        return _SORTED_LLM_REGISTRY

    # ── Facade: Strategy dispatch ─────────────────────────────────────────────

    def invoke(
        self, *, model_id: str, api_key: str | None = None, **kwargs: Any
    ) -> Any:
        """Resolve the correct provider strategy and invoke a chat completion."""
        return self.get_provider(model_id, api_key).invoke(**kwargs)

    def stream(
        self, *, model_id: str, api_key: str | None = None, **kwargs: Any
    ) -> Generator[tuple[str | None, str | None], None, None]:
        """Resolve the correct provider strategy and stream a chat completion."""
        yield from self.get_provider(model_id, api_key).stream(**kwargs)

    def get_provider(
        self,
        model_id: str,
        api_key: str | None = None,
    ) -> LLMProvider:
        """Return an instantiated provider for model_id, validated against the registry.

        Raises ValueError for unknown model IDs. Providers are stateless — a fresh
        instance per call is intentional so per-call kwargs (reasoning_effort, timeout)
        can vary across invocations without affecting cached state.
        """
        if LLMProvider.get_entry(model_id) is None:
            raise ValueError(f"Model ID '{model_id}' not found in registry.")
        provider_cls, resolved_key = self._resolve_cls_and_key(model_id, api_key)
        return provider_cls(model_id=model_id, api_key=resolved_key)


# Module-level singleton
llm_manager = LLMManager()


def get_llm(model_id: str, api_key: str | None = None) -> LLMProvider:
    """Return an instantiated LLMProvider for model_id, validated against the registry."""
    return llm_manager.get_provider(model_id, api_key)
