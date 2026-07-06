"""Abstract base class for embedding providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from app.core.models.embedding.registry import (
    EMBEDDING_REGISTRY,
    EMBEDDING_REGISTRY_IDS,
    EmbeddingModelEntry,
    EmbeddingProviderName,
)

__all__ = [
    "EmbeddingProvider",
    "EmbeddingModelEntry",
    "EmbeddingProviderName",
    "EMBEDDING_REGISTRY",
    "EMBEDDING_REGISTRY_IDS",
]

_REGISTRY_BY_ID: dict[str, EmbeddingModelEntry] = {e.id: e for e in EMBEDDING_REGISTRY}

# Populated by subclass definitions via __init_subclass__
_PROVIDER_CLASS_MAP: dict[EmbeddingProviderName, type[EmbeddingProvider]] = {}


class EmbeddingProvider(ABC):
    """Abstract base for all embedding providers.

    Subclasses declare PROVIDER_NAME and implement build() + api_key_from_settings().
    Declaring PROVIDER_NAME auto-registers the subclass so EmbeddingManager can
    resolve the correct class without a separate mapping.

    EmbeddingManager owns caching — subclasses are stateless builders.
    """

    PROVIDER_NAME: ClassVar[EmbeddingProviderName]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "PROVIDER_NAME"):
            _PROVIDER_CLASS_MAP[cls.PROVIDER_NAME] = cls

    def __init__(self, model_id: str, api_key: str, dimensions: int | None = None, **kwargs: Any) -> None:
        self.model_id = model_id
        self.api_key = api_key
        self.dimensions = dimensions

    @abstractmethod
    def build(self) -> Any:
        """Construct and return a LlamaIndex BaseEmbedding instance."""

    @classmethod
    @abstractmethod
    def api_key_from_settings(cls, settings: Any) -> str:
        """Return the API key for this provider from the settings object."""

    @staticmethod
    def for_provider(provider: EmbeddingProviderName) -> type[EmbeddingProvider]:
        """Return the provider class registered for the given provider name."""
        cls = _PROVIDER_CLASS_MAP.get(provider)
        if cls is None:
            raise ValueError(
                f"No EmbeddingProvider registered for '{provider}'. "
                f"Define a subclass with PROVIDER_NAME = '{provider}'."
            )
        return cls

    @staticmethod
    def detect_provider(model_id: str) -> EmbeddingProviderName:
        """Resolve provider via registry lookup → prefix heuristic."""
        entry = _REGISTRY_BY_ID.get(model_id)
        if entry is not None:
            return entry.provider
        m = model_id.lower()
        if m.startswith("gemini-") or m.startswith("models/gemini"):
            return "gemini"
        return "openai"

    @staticmethod
    def get_entry(model_id: str) -> EmbeddingModelEntry | None:
        return _REGISTRY_BY_ID.get(model_id)
