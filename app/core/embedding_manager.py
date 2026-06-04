"""Singleton factory and cache for LlamaIndex embedding instances."""
from __future__ import annotations

from threading import Lock
from typing import Any

from app.core.config import settings
from app.core.models.embedding.base import EMBEDDING_REGISTRY, EmbeddingModelEntry, EmbeddingProvider
# Side-effect imports: register provider subclasses via __init_subclass__
import app.core.models.embedding.openai  # noqa: F401
import app.core.models.embedding.gemini  # noqa: F401

_PROVIDER_SORT_ORDER: dict[str, int] = {"openai": 0, "gemini": 1}
_SORTED_REGISTRY: list[EmbeddingModelEntry] = sorted(
    EMBEDDING_REGISTRY, key=lambda e: _PROVIDER_SORT_ORDER.get(e.provider, 2)
)


class EmbeddingManager:
    """Factory and thread-safe cache for LlamaIndex BaseEmbedding instances.

    Keyed by (model_id, api_key, dimensions) — the same combination always
    returns the same object, avoiding redundant initialisation across concurrent
    workers.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str | None, int | None], Any] = {}
        self._lock = Lock()

    def get_instance(
        self, *, model_id: str, api_key: str | None = None, dimensions: int | None = None
    ) -> Any:
        """Return a cached or freshly built embedding instance."""
        provider_cls = EmbeddingProvider.for_provider(EmbeddingProvider.detect_provider(model_id))
        resolved_key = api_key or provider_cls.api_key_from_settings(settings)
        cache_key = (model_id, resolved_key, dimensions)
        with self._lock:
            if cache_key not in self._cache:
                self._cache[cache_key] = self._build(
                    provider_cls=provider_cls,
                    model_id=model_id,
                    api_key=resolved_key,
                    dimensions=dimensions,
                )
            return self._cache[cache_key]

    def _build(
        self,
        *,
        provider_cls: type[EmbeddingProvider],
        model_id: str,
        api_key: str,
        dimensions: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        return provider_cls(model_id=model_id, api_key=api_key, **kwargs).build()

    def invalidate(self, model_id: str | None = None) -> None:
        """Evict one model's cached instance, or all when model_id is None."""
        with self._lock:
            if model_id is None:
                self._cache.clear()
            else:
                self._cache = {k: v for k, v in self._cache.items() if k[0] != model_id}

    def list_models(self) -> list[EmbeddingModelEntry]:
        """Return registry entries sorted openai → gemini → other."""
        return _SORTED_REGISTRY


# Module-level singleton — imported by get_embeddings() and initialize_ai_provider()
embedding_manager = EmbeddingManager()
