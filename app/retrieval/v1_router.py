"""LlamaIndex-backed strategy selector for the traditional v1 RAG path."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import ToolMetadata

from app.core.ai_provider import get_llm
from app.core.config import settings


class RetrievalStrategy(StrEnum):
    """Candidate-generation strategies supported by traditional retrieval."""

    DIRECT = "direct"
    EXPANDED = "expanded"
    DECOMPOSED = "decomposed"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Selected strategy and the selector's concise rationale."""

    strategy: RetrievalStrategy
    reason: str


_ROUTE_CHOICES: tuple[ToolMetadata, ...] = (
    ToolMetadata(
        name=RetrievalStrategy.DIRECT.value,
        description=(
            "Use for a precise single-fact, single-metric, or numeric lookup that can "
            "be retrieved directly from one focused query."
        ),
    ),
    ToolMetadata(
        name=RetrievalStrategy.EXPANDED.value,
        description=(
            "Use for an ambiguous, broad, summary, visual, or conversational follow-up "
            "query that benefits from alternate retrieval wording."
        ),
    ),
    ToolMetadata(
        name=RetrievalStrategy.DECOMPOSED.value,
        description=(
            "Use for compound, comparative, temporal, multi-entity, or multi-metric "
            "questions that require several independently retrievable sub-queries."
        ),
    ),
)


class V1RetrievalRouter:
    """Select one v1 retrieval strategy without owning query execution."""

    def __init__(self, selector: PydanticSingleSelector):
        self._selector = selector

    def route(self, question: str) -> RoutingDecision:
        result = self._selector.select(_ROUTE_CHOICES, question)
        selected = _ROUTE_CHOICES[result.ind]
        return RoutingDecision(
            strategy=RetrievalStrategy(selected.name),
            reason=str(result.reason or "").strip(),
        )


@lru_cache(maxsize=4)
def get_v1_retrieval_router(model: str | None = None) -> V1RetrievalRouter:
    """Return a cached router for the configured low-cost query model."""
    model_id = model or settings.V1_QUERY_ROUTER_MODEL or settings.QUERY_EXPANSION_MODEL
    selector = PydanticSingleSelector.from_defaults(
        llm=get_llm(
            model=model_id,
            reasoning_effort="none",
            timeout_seconds=15,
        )
    )
    return V1RetrievalRouter(selector)
