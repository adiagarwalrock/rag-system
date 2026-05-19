"""Provider-specific factories for LLM and embedding construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from google.genai import types as genai_types
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI, OpenAIResponses


class SupportedReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


DEFAULT_REASONING_EFFORT = SupportedReasoningEffort.MEDIUM
SUPPORTED_REASONING_EFFORTS = SupportedReasoningEffort


class AIProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


_PLACEHOLDER_API_KEYS = {
    "your_openai_api_key_here",
    "your_api_key_here",
    "your_api_key",
    "your_ai_api_key",
    "your_gemini_api_key_here",
    "your_google_api_key_here",
}

_MESSAGE_ROLE_MAP: dict[str, MessageRole] = {
    "system": MessageRole.SYSTEM,
    "developer": MessageRole.DEVELOPER,
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    "tool": MessageRole.TOOL,
    "function": MessageRole.FUNCTION,
    "model": MessageRole.MODEL,
    "chatbot": MessageRole.CHATBOT,
}

_GEMINI_THINKING_BUDGET_BY_EFFORT = {
    "low": 512,
    "medium": 2048,
    "high": 8192,
}


def normalize_api_key(value: str | None) -> str:
    key = (value or "").strip()
    return key.strip("'\"").strip()


def is_placeholder_api_key(value: str | None) -> bool:
    key = normalize_api_key(value).lower()
    return not key or key in _PLACEHOLDER_API_KEYS or key.startswith("your_")


def normalize_reasoning_effort(
    reasoning_effort: str | None,
) -> SupportedReasoningEffort:
    effort = (reasoning_effort or "").strip().lower()
    try:
        return SupportedReasoningEffort(effort)
    except ValueError:
        return DEFAULT_REASONING_EFFORT


class BaseAIProviderFactory(ABC):
    provider_name: AIProvider

    @abstractmethod
    def create_llm(
        self,
        *,
        model: str,
        api_key: str,
        reasoning_effort: str | None,
        use_responses_api: bool,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def create_embedding(
        self,
        *,
        model: str,
        api_key: str,
        output_dimension: int | None,
    ) -> Any:
        raise NotImplementedError

    def to_chat_messages(
        self, input_messages: list[dict[str, Any]]
    ) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        for message in input_messages:
            if not isinstance(message, dict):
                continue
            role = self._resolve_message_role(message.get("role"))
            content = message.get("content")
            blocks = self._content_to_blocks(content)
            if blocks is not None:
                messages.append(ChatMessage(role=role, blocks=blocks))
                continue
            messages.append(ChatMessage(role=role, content=str(content or "")))

        if not messages:
            raise ValueError("input_messages must include at least one message")
        return messages

    @abstractmethod
    def build_chat_runtime_kwargs(
        self,
        *,
        max_output_tokens: int | None,
        prompt_cache_key: str | None,
        prompt_cache_retention: str | None,
        safety_identifier: str | None,
        user_tag: str | None,
        timeout_seconds: float | None,
        use_responses_api: bool,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _resolve_message_role(self, value: Any) -> MessageRole:
        role = str(value or "user").strip().lower()
        return _MESSAGE_ROLE_MAP.get(role, MessageRole.USER)

    @staticmethod
    def _content_to_blocks(content: Any) -> list[Any] | None:
        if not isinstance(content, list):
            return None

        blocks: list[Any] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type == "input_text":
                text = str(item.get("text") or "").strip()
                if text:
                    blocks.append(TextBlock(text=text))
                continue
            if item_type == "input_image":
                image_url = str(item.get("image_url") or "").strip()
                if image_url:
                    blocks.append(ImageBlock(url=image_url))

        return blocks or None


class OpenAIProviderFactory(BaseAIProviderFactory):
    provider_name = AIProvider.OPENAI

    def create_llm(
        self,
        *,
        model: str,
        api_key: str,
        reasoning_effort: str | None,
        use_responses_api: bool,
    ) -> Any:
        llm_class = OpenAIResponses if use_responses_api else OpenAI
        kwargs: dict[str, Any] = {"model": model, "api_key": api_key}
        if llm_class is OpenAIResponses and reasoning_effort is not None:
            kwargs["reasoning_options"] = {
                "effort": normalize_reasoning_effort(reasoning_effort)
            }
        return llm_class(**kwargs)

    def create_embedding(
        self,
        *,
        model: str,
        api_key: str,
        output_dimension: int | None,
    ) -> Any:
        embedding_kwargs: dict[str, Any] = {}
        if output_dimension is not None:
            embedding_kwargs["dimensions"] = output_dimension
        return OpenAIEmbedding(
            model=model,
            api_key=api_key,
            **embedding_kwargs,
        )

    def build_chat_runtime_kwargs(
        self,
        *,
        max_output_tokens: int | None,
        prompt_cache_key: str | None,
        prompt_cache_retention: str | None,
        safety_identifier: str | None,
        user_tag: str | None,
        timeout_seconds: float | None,
        use_responses_api: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if timeout_seconds is not None:
            kwargs["timeout"] = float(timeout_seconds)
        if user_tag:
            kwargs["user"] = user_tag

        if use_responses_api:
            kwargs["truncation"] = "disabled"
            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens
            return kwargs

        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        return kwargs


class GeminiProviderFactory(BaseAIProviderFactory):
    provider_name = AIProvider.GEMINI

    def create_llm(
        self,
        *,
        model: str,
        api_key: str,
        reasoning_effort: str | None,
        use_responses_api: bool,
    ) -> Any:
        kwargs: dict[str, Any] = {"model": model, "api_key": api_key}
        if reasoning_effort is not None:
            kwargs["generation_config"] = self._build_reasoning_generation_config(
                model=model,
                reasoning_effort=reasoning_effort,
            )
        return GoogleGenAI(**kwargs)

    def create_embedding(
        self,
        *,
        model: str,
        api_key: str,
        output_dimension: int | None,
    ) -> Any:
        kwargs: dict[str, Any] = {"model_name": model, "api_key": api_key}
        if output_dimension is not None:
            kwargs["embedding_config"] = {
                "output_dimensionality": output_dimension,
            }
        return GoogleGenAIEmbedding(**kwargs)

    def build_chat_runtime_kwargs(
        self,
        *,
        max_output_tokens: int | None,
        prompt_cache_key: str | None,
        prompt_cache_retention: str | None,
        safety_identifier: str | None,
        user_tag: str | None,
        timeout_seconds: float | None,
        use_responses_api: bool,
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {}
        if max_output_tokens is not None:
            generation_config["max_output_tokens"] = max_output_tokens

        cached_content = self._normalize_cached_content(prompt_cache_key)
        if cached_content:
            generation_config["cached_content"] = cached_content

        kwargs: dict[str, Any] = {}
        if generation_config:
            kwargs["generation_config"] = generation_config

        if safety_identifier:
            kwargs["safety_settings"] = [
                genai_types.SafetySetting(
                    category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                )
            ]

        if timeout_seconds is not None:
            # Map timeout to GoogleGenAI's request_options.timeout
            kwargs["request_options"] = {"timeout": timeout_seconds}

        # user_tag can be mapped if LlamaIndex updates their Gemini integration
        # For now we'll pass it, but invoke_llm_chat may pop it for safety
        if user_tag:
            kwargs["user"] = user_tag

        return kwargs

    def _resolve_message_role(self, value: Any) -> MessageRole:
        role = super()._resolve_message_role(value)
        if role == MessageRole.DEVELOPER:
            return MessageRole.SYSTEM
        return role

    @staticmethod
    def _build_reasoning_generation_config(
        *,
        model: str,
        reasoning_effort: str | SupportedReasoningEffort,
    ) -> genai_types.GenerateContentConfig:
        effort = normalize_reasoning_effort(reasoning_effort)
        normalized_model = model.lower()
        if "gemini-2.5" in normalized_model:
            return genai_types.GenerateContentConfig(
                thinking_config=genai_types.ThinkingConfig(
                    thinking_budget=_GEMINI_THINKING_BUDGET_BY_EFFORT[effort]
                )
            )

        return genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(thinking_level=effort)
        )

    @staticmethod
    def _normalize_cached_content(prompt_cache_key: str | None) -> str | None:
        value = (prompt_cache_key or "").strip()
        if not value:
            return None
        if value.startswith("projects/") and "/cachedContents/" in value:
            return value
        return None


class AIProviderFactoryResolver:
    """Resolve the active provider and expose its concrete factory."""

    def __init__(
        self,
        *,
        openai_api_key: str | None,
        gemini_api_key: str | None,
        google_api_key: str | None,
    ) -> None:
        self.openai_api_key = normalize_api_key(openai_api_key)
        self.gemini_api_key = normalize_api_key(gemini_api_key)
        self.google_api_key = normalize_api_key(google_api_key)

    def resolve_provider(self, *, model: str | None) -> AIProvider:
        if self.gemini_api_key or self.google_api_key:
            return AIProvider.GEMINI
        if self.openai_api_key:
            return AIProvider.OPENAI

        model_name = (model or "").strip().lower()
        if "gemini" in model_name:
            return AIProvider.GEMINI
        return AIProvider.OPENAI

    def resolve_api_key(self, *, provider: AIProvider, fallback_key: str | None) -> str:
        if provider == AIProvider.GEMINI:
            return (
                self.gemini_api_key
                or self.google_api_key
                or normalize_api_key(fallback_key)
            )
        return self.openai_api_key or normalize_api_key(fallback_key)

    @staticmethod
    def get_factory(provider: AIProvider) -> BaseAIProviderFactory:
        if provider == AIProvider.GEMINI:
            return GeminiProviderFactory()
        return OpenAIProviderFactory()
