"""OpenAI embedding provider."""
from __future__ import annotations

from typing import Any

from llama_index.embeddings.openai import OpenAIEmbedding

from app.core.models.embedding.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    PROVIDER_NAME = "openai"

    def build(self) -> Any:
        extra: dict[str, Any] = {"dimensions": self.dimensions} if self.dimensions is not None else {}
        return OpenAIEmbedding(model=self.model_id, api_key=self.api_key, **extra)

    @classmethod
    def api_key_from_settings(cls, settings: Any) -> str:
        return settings.openai_api_key
