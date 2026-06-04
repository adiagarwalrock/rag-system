"""Embedding model registry — the single source of truth for known models.

To add a model: append one EmbeddingModelEntry line to EMBEDDING_REGISTRY.
Unknown model IDs still work at runtime via prefix detection in EmbeddingProvider;
a registry entry is only needed to appear in the UI list and carry dimension metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EmbeddingProviderName = Literal["openai", "gemini"]


@dataclass(frozen=True)
class EmbeddingModelEntry:
    id: str
    provider: EmbeddingProviderName
    dimensions: int
    display_name: str = ""


EMBEDDING_REGISTRY: list[EmbeddingModelEntry] = [
    # ── OpenAI ────────────────────────────────────────────────────────────────
    # text-embedding-3-large: 3072d default, supports dimension reduction, 8k ctx
    EmbeddingModelEntry(
        "text-embedding-3-large", "openai", 3072, "OpenAI text-embedding-3-large"
    ),
    # text-embedding-3-small: 1536d default, lower cost, 8k ctx
    EmbeddingModelEntry(
        "text-embedding-3-small", "openai", 1536, "OpenAI text-embedding-3-small"
    ),
    # text-embedding-ada-002: previous generation, 1536d, 8k ctx
    EmbeddingModelEntry(
        "text-embedding-ada-002", "openai", 1536, "OpenAI Ada 002 (legacy)"
    ),
    # ── Gemini ────────────────────────────────────────────────────────────────
    # gemini-embedding-2: multimodal (text/image/video/audio/PDF), up to 3072d, 8k ctx
    EmbeddingModelEntry(
        "gemini-embedding-2", "gemini", 3072, "Gemini Embedding 2 (multimodal)"
    ),
    # gemini-embedding-001: text-only, up to 3072d, 2k ctx
    EmbeddingModelEntry(
        "gemini-embedding-001", "gemini", 3072, "Gemini Embedding 001 (text)"
    ),
]

EMBEDDING_REGISTRY_IDS: frozenset[str] = frozenset(e.id for e in EMBEDDING_REGISTRY)
