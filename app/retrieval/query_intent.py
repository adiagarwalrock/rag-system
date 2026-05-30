"""
Lightweight retrieval intent detection for evidence planning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "BXP": ("bxp", "boston properties"),
    "Digital Realty": ("digital realty", "dlr"),
    "Public Storage": ("public storage", "psa"),
    "Realty Income": ("realty income",),
    "VICI": ("vici",),
    "EastGroup": ("eastgroup", "egp"),
    "Simon": ("simon", "simon property"),
}


@dataclass(frozen=True, slots=True)
class RetrievalIntent:
    labels: tuple[str, ...] = ()
    companion_queries: tuple[str, ...] = ()
    entities: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def has(self, label: str) -> bool:
        return label in self.labels


def analyze_retrieval_intent(question: str) -> RetrievalIntent:
    normalized = _normalize(question)
    labels: list[str] = []
    companions: list[str] = []
    entities = query_entity_aliases(normalized)

    if _is_temporal_delta(normalized):
        labels.append("temporal_delta")
        companions.extend(
            (
                f"{question} older prior investor day plan strategy action plan",
                f"{question} newer latest quarterly update q4 changes guidance",
            )
        )

    if _is_stale_source_sensitive(normalized):
        labels.append("stale_source")

    if _is_caveat_or_inconsistency_sensitive(normalized):
        labels.append("caveat_inconsistency")
        companions.extend(
            (
                f"{question} footnote as of date caveat inconsistency",
                f"{question} headline appendix table same metric",
            )
        )

    if _is_outlook_scope_sensitive(normalized):
        labels.append("outlook_scope")
        companions.extend(
            (
                f"{question} guidance outlook pro forma merger acquisition",
                f"{question} ffo neutral accretive stabilization standalone",
                f"{question} investor day prior year historical presentation baseline",
            )
        )

    if len(entities) >= 2:
        labels.append("named_entity_comparison")
        for entity_name in entities:
            companions.append(f"{question} {entity_name}")

    if _needs_balanced_scope(normalized):
        labels.append("balanced_scope")

    return RetrievalIntent(
        labels=tuple(dict.fromkeys(labels)),
        companion_queries=tuple(_dedupe_queries(companions, question)),
        entities=entities,
    )


def query_entity_aliases(normalized_question: str) -> dict[str, tuple[str, ...]]:
    matches: dict[str, tuple[str, ...]] = {}
    for entity_name, aliases in ENTITY_ALIASES.items():
        if any(_contains_alias(normalized_question, alias) for alias in aliases):
            matches[entity_name] = aliases
    return matches


def _contains_alias(normalized_question: str, alias: str) -> bool:
    if " " in alias:
        return alias in normalized_question
    return (
        re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized_question)
        is not None
    )


def _is_temporal_delta(normalized: str) -> bool:
    has_delta = any(
        term in normalized
        for term in (
            "what changed",
            "what's changed",
            "change between",
            "changed between",
            "between",
            "versus",
            " vs ",
            "compare",
        )
    )
    has_temporal_anchor = any(
        term in normalized
        for term in (
            "investor day",
            "q4",
            "quarterly update",
            "update",
            "older",
            "newer",
            "previous",
            "latest",
            "2025",
            "2026",
        )
    )
    if has_delta and has_temporal_anchor:
        return True
    # Projection/forecast questions implicitly span multiple documents: a projected figure
    # from an Investor Day and the same metric updated in a later quarterly deck are both
    # relevant answers. Treat "projected/forecast/expected + year" as temporal_delta so
    # _ensure_temporal_delta_evidence fires and pulls both documents into evidence.
    has_projection = any(
        term in normalized for term in ("projected", "forecast", "expected")
    )
    return has_projection and has_temporal_anchor


def _is_stale_source_sensitive(normalized: str) -> bool:
    return any(
        term in normalized
        for term in (
            "economic impact",
            "local communities",
            "community impact",
            "jobs",
            "tax revenue",
        )
    )


def _is_caveat_or_inconsistency_sensitive(normalized: str) -> bool:
    if any(
        term in normalized
        for term in (
            "inconsistency",
            "inconsistent",
            "contradict",
            "conflict",
            "caveat",
            "footnote",
            "changed between",
            "change between",
            "changed from",
            "how has",
            "methodology",
            "basis changed",
        )
    ):
        return True
    return any(
        term in normalized for term in ("how many", "number of", "count")
    ) and any(
        subject in normalized
        for subject in (
            "customers",
            "countries",
            "properties",
            "markets",
            "assets",
        )
    )


def _is_outlook_scope_sensitive(normalized: str) -> bool:
    return any(
        term in normalized
        for term in (
            "outlook",
            "guidance",
            "forecast",
            "projection",
            "projected",
            "2026 ffo",
            "ffo outlook",
            "ffo/share",
        )
    )


def _needs_balanced_scope(normalized: str) -> bool:
    return any(
        term in normalized
        for term in (
            "different",
            "both",
            "each",
            "across",
            "sectors",
            "represented",
            "by reit",
            "for each",
            "in the corpus",
            "corpus",
            "for each reit",
            "each company",
            "each reit",
            "per reit",
            "per company",
            "all reits",
        )
    )


def _dedupe_queries(candidates: list[str], original_question: str) -> list[str]:
    original_normalized = _normalize(original_question)
    seen = {original_normalized}
    deduped: list[str] = []
    for candidate in candidates:
        cleaned = " ".join(candidate.split())
        normalized = _normalize(cleaned)
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(cleaned)
    return deduped[:6]


def _normalize(value: str) -> str:
    return " ".join(str(value or "").lower().split())
