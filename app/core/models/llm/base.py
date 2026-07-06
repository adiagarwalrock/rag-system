"""Abstract base class for LLM providers.

Patterns:
- Factory Method: build_llm() lets each subclass decide which LlamaIndex LLM object
  to produce. Only OpenAI and Gemini implement this — Anthropic doesn't wrap a
  LlamaIndex object, so it inherits the default NotImplementedError.
- Strategy: invoke() and stream() are the strategy interface. Each subclass
  encapsulates its own invocation algorithm. Callers never branch on provider name.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from app.core.models.llm.registry import (
    LLM_REGISTRY,
    LLMModelEntry,
    LLMProviderName,
)

_S = TypeVar("_S", bound=BaseModel)

_REGISTRY_BY_ID: dict[str, LLMModelEntry] = {e.id: e for e in LLM_REGISTRY}

# Populated by subclass definitions via __init_subclass__
_LLM_PROVIDER_CLASS_MAP: dict[LLMProviderName, type[LLMProvider]] = {}


class LLMProvider(ABC):
    """Abstract base for all LLM providers.

    Each subclass is both a Concrete Strategy (invoke/stream algorithms) and,
    optionally, a Factory Method implementor (build_llm for LlamaIndex LLM objects).

    LLMManager owns caching and acts as the Facade — subclasses are stateless.
    """

    PROVIDER_NAME: ClassVar[LLMProviderName]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "PROVIDER_NAME"):
            _LLM_PROVIDER_CLASS_MAP[cls.PROVIDER_NAME] = cls

    def __init__(self, model_id: str, api_key: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.api_key = api_key

    # ── Factory Method ────────────────────────────────────────────────────────

    def build_llm(self, **kwargs: Any) -> Any:
        """Return a LlamaIndex-compatible LLM instance.

        Used by get_llm() and LlamaIndex global settings. kwargs are forwarded
        to providers that support them (e.g. reasoning_effort for Anthropic,
        reasoning_effort/reasoning_summary/timeout_seconds for OpenAI).
        Providers that don't support a kwarg silently ignore it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a LlamaIndex LLM object. "
            "Use invoke() / stream() directly."
        )

    # ── Strategy interface ────────────────────────────────────────────────────

    @abstractmethod
    def invoke(
        self,
        input_messages: list[dict[str, Any]],
        *,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        max_output_tokens: int | None = None,
        timeout_seconds: float | None = None,
        structured_output_schema: type[_S] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke a chat completion and return a response.

        input_messages is the current turn(s). Providers support multi-turn history
        naturally — pass the full conversation as a list of role-tagged dicts.
        Image content is passed via list-typed message content with type=input_image.
        Return value must be compatible with extract_chat_response_text().
        Provider-specific extras (prompt_cache_key, safety_identifier, etc.)
        are accepted via **kwargs and silently ignored by providers that don't
        support them.
        """

    @abstractmethod
    def stream(
        self,
        input_messages: list[dict[str, Any]],
        *,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        max_output_tokens: int | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> Generator[tuple[str | None, str | None], None, None]:
        """Stream a chat completion, yielding (reasoning_delta, answer_delta) tuples.

        reasoning_delta is non-None only when the model emits a thinking/reasoning block.
        answer_delta carries normal response tokens.
        """

    # ── Registry helpers ──────────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def api_key_from_settings(cls, settings: Any) -> str:
        """Return the API key for this provider from the settings object."""

    @staticmethod
    def for_provider(provider: LLMProviderName) -> type[LLMProvider]:
        """Return the provider class registered for the given provider name."""
        cls = _LLM_PROVIDER_CLASS_MAP.get(provider)
        if cls is None:
            raise ValueError(
                f"No LLMProvider registered for '{provider}'. "
                f"Define a subclass with PROVIDER_NAME = '{provider}'."
            )
        return cls

    @staticmethod
    def detect_provider(model_id: str) -> LLMProviderName:
        """Resolve provider via registry lookup → prefix heuristic.

        Bare model names (no '/') are assumed to be OpenAI for backward compatibility.
        """
        entry = _REGISTRY_BY_ID.get(model_id)
        if entry is not None:
            return entry.provider
        prefix = model_id.split("/")[0].lower() if "/" in model_id else ""
        if prefix == "anthropic":
            return "anthropic"
        if prefix in {"gemini", "vertex_ai"}:
            return "gemini"
        return "openai"

    @staticmethod
    def get_entry(model_id: str) -> LLMModelEntry | None:
        return _REGISTRY_BY_ID.get(model_id)
