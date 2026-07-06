"""LLM model registry — single source of truth for known LLM models.

To add a model: append one LLMModelEntry line to LLM_REGISTRY.
Model IDs use the LiteLLM convention: "provider/model-name".
OpenAI models can also be referenced with bare names (e.g. "gpt-5.5");
the dispatch layer treats any model without a "/" as OpenAI.

SOC2 note: all listed providers are SOC2 Type 2 certified.
For Gemini in strict enterprise environments, replace "gemini/" prefix
with "vertex_ai/" (Google Vertex AI Enterprise) in your LLM_MODEL setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LLMProviderName = Literal["openai", "anthropic", "gemini"]


@dataclass(frozen=True)
class LLMModelEntry:
    id: str  # LiteLLM model string e.g. "anthropic/claude-opus-4-8"
    provider: LLMProviderName
    display_name: str
    context_window: int
    supports_reasoning: bool  # reasoning_effort="low|medium|high" supported
    supports_vision: bool
    default: bool = field(default=False)


LLM_REGISTRY: list[LLMModelEntry] = [
    # ── OpenAI ───────────────────────────────────────────────
    LLMModelEntry("openai/gpt-5.5", "openai", "GPT-5.5", 1_050_000, True, True, True),
    LLMModelEntry("openai/gpt-5.5-pro", "openai", "GPT-5.5 pro", 1_050_000, True, True),
    LLMModelEntry("openai/gpt-5.4-mini", "openai", "GPT-5.4 mini", 400_000, True, True),
    LLMModelEntry("openai/gpt-5.4-nano", "openai", "GPT-5.4 nano", 400_000, True, True),
    # ── Anthropic ────────────────────────────────────────────
    LLMModelEntry(
        "anthropic/claude-opus-4-8",
        "anthropic",
        "Claude Opus 4.8",
        1_000_000,
        True,
        True,
    ),
    LLMModelEntry(
        "anthropic/claude-sonnet-4-6",
        "anthropic",
        "Claude Sonnet 4.6",
        1_000_000,
        True,
        True,
    ),
    LLMModelEntry(
        "anthropic/claude-haiku-4-5",
        "anthropic",
        "Claude Haiku 4.5",
        200_000,
        True,
        True,
    ),
    # ── Gemini ───────────────────────────────────────────────
    LLMModelEntry(
        "gemini/gemini-3.1-pro-preview",
        "gemini",
        "Gemini 3.1 Pro Preview",
        1_048_576,
        True,
        True,
    ),
    LLMModelEntry(
        "gemini/gemini-3.5-flash", "gemini", "Gemini 3.5 Flash", 1_048_576, True, True
    ),
    LLMModelEntry(
        "gemini/gemini-3.1-flash-lite",
        "gemini",
        "Gemini 3.1 Flash-Lite",
        1_048_576,
        True,
        True,
    ),
]

LLM_REGISTRY_IDS: frozenset[str] = frozenset(e.id for e in LLM_REGISTRY)
