"""
Lightweight retrieval intent detection for evidence planning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalIntent:
    labels: tuple[str, ...] = ()
    companion_queries: tuple[str, ...] = ()

    def has(self, label: str) -> bool:
        return label in self.labels


def analyze_retrieval_intent(question: str) -> RetrievalIntent:
    normalized = _normalize(question)
    labels: list[str] = []
    companions: list[str] = []

    if _is_temporal_delta(normalized):
        labels.append("temporal_delta")
        companions.extend(
            (
                f"{question} older prior presentation annual report baseline strategy",
                f"{question} newer latest quarterly update changes guidance",
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
                f"{question} neutral accretive stabilization standalone",
                f"{question} prior year presentation historical baseline estimate",
            )
        )

    if _needs_balanced_scope(normalized):
        labels.append("balanced_scope")

    return RetrievalIntent(
        labels=tuple(dict.fromkeys(labels)),
        companion_queries=tuple(_dedupe_queries(companions, question)),
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
            "q4",
            "q1",
            "q2",
            "q3",
            "quarterly",
            "quarterly update",
            "annual",
            "update",
            "older",
            "newer",
            "previous",
            "latest",
            "prior",
        )
    ) or bool(re.search(r"\b(?:19|20)\d{2}\b", normalized))
    if has_delta and has_temporal_anchor:
        return True
    # Projection/forecast questions implicitly span multiple documents: a projected figure
    # from a prior presentation and the same metric updated in a later filing are both
    # relevant. Treat "projected/forecast/expected + temporal anchor" as temporal_delta.
    has_projection = any(
        term in normalized for term in ("projected", "forecast", "expected")
    )
    return has_projection and has_temporal_anchor


def _is_stale_source_sensitive(normalized: str) -> bool:
    """Detect questions about community/social/economic outputs whose source date matters."""
    has_impact_signal = any(
        term in normalized for term in ("impact", "contribution", "effect")
    )
    has_community_signal = any(
        term in normalized
        for term in (
            "communities",
            "community",
            "jobs",
            "employment",
            "tax",
            "local",
            "regional",
        )
    )
    return has_impact_signal and has_community_signal


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
            "locations",
            "tenants",
            "units",
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
            "for each",
            "each company",
            "each issuer",
            "per company",
            "per issuer",
            "all companies",
            "all issuers",
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
