"""Gemini embedding provider."""
from __future__ import annotations

from typing import Any

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from app.core.models.embedding.base import EmbeddingProvider


class GeminiEmbeddingProvider(EmbeddingProvider):
    PROVIDER_NAME = "gemini"

    def build(self) -> Any:
        # GoogleGenAIEmbedding uses model_name kwarg, not model
        return GoogleGenAIEmbedding(model_name=self.model_id, api_key=self.api_key)

    @classmethod
    def api_key_from_settings(cls, settings: Any) -> str:
        return settings.gemini_api_key
