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
        # Use short, focused retrieval queries (not the full question repeated) so both
        # document versions actually rank well in the candidate pool.
        _add_temporal_delta_companions(question, companions)

    if _is_stale_source_sensitive(normalized):
        labels.append("stale_source")

    if _is_caveat_or_inconsistency_sensitive(normalized):
        labels.append("caveat_inconsistency")
        companions.extend(
            (
                f"{question} footnote as of date caveat inconsistency",
                f"{question} headline appendix table same metric",
                f"{question} definition scope qualifier basis",
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

    if _is_multi_version_lookup(normalized):
        labels.append("multi_version_lookup")

    if _is_visual_detail_lookup(normalized):
        labels.append("visual_detail")
        companions.extend(_visual_detail_companions(question))

    if _is_corpus_wide_scope(normalized):
        labels.append("corpus_wide_scope")
        # Generate broad companion queries so every document contributes to the pool.
        companions.extend(_corpus_wide_companions(question))

    return RetrievalIntent(
        labels=tuple(dict.fromkeys(labels)),
        companion_queries=tuple(_dedupe_queries(companions, question)),
    )


def _add_temporal_delta_companions(question: str, companions: list[str]) -> None:
    """Add short standalone companion queries to surface both old and new document versions.

    Uses the core metric/topic phrase (with comparison framing stripped) so that both
    the baseline and updated versions score well on the specific metric being asked about,
    rather than matching only on the comparison framing words.
    """
    topic = _extract_topic_phrase(question)
    companions.extend(
        (
            f"{topic} investor day baseline presentation",
            f"{topic} quarterly update current guidance",
        )
    )


def _extract_topic_phrase(question: str) -> str:
    """Strip 'what changed between X and Y' framing, return the core metric/topic."""
    q = re.sub(
        r"\bwhat(?:'s)?\s+changed\b.*?\bbetween\b\s*",
        "",
        question,
        flags=re.IGNORECASE,
    )
    q = re.sub(
        r"\bbetween\s+the\s+.*?\band\s+the\b.*$",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = q.strip()
    return q if len(q) > 10 else question


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
        _contains_term(normalized, term)
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
            "actual market",
        )
    ):
        return True
    if "as of" in normalized and any(
        metric in normalized
        for metric in (
            "yield",
            "margin",
            "occupancy",
            "customers",
            "countries",
            "portfolio",
            "square feet",
            "rent",
            "noi",
        )
    ):
        return True
    if any(term in normalized for term in ("what percentage", "what share")) and any(
        subject in normalized
        for subject in (
            "portfolio",
            "rent",
            "noi",
            "markets",
            "square feet",
            "base rent",
        )
    ):
        return True
    if "how big" in normalized and any(
        subject in normalized for subject in ("portfolio", "square feet", "owned")
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


def _is_multi_version_lookup(normalized: str) -> bool:
    """Detect simple factual lookups where multiple document versions likely carry the answer.

    For questions like "What is X's dividend yield?" or "What is X's strategy?", if the
    corpus contains multiple presentations from the same issuer, the answer may differ across
    versions.  Flagging this intent lets evidence selection guarantee both versions are pulled.
    """
    # Multi-party/breadth questions need balanced retrieval, not same-issuer version quotas.
    if any(
        term in normalized
        for term in (
            " between ",
            " vs ",
            " versus ",
            "compare",
            "difference",
            "each company",
            "each reit",
            "among ",
            "across ",
            " in the corpus",
        )
    ):
        return False

    # Metric/stat lookups that are typically present in multiple periodic investor updates.
    has_simple_lookup = any(
        term in normalized
        for term in (
            "what is",
            "what are",
            "what was",
            "what does",
            "what did",
            "what percentage",
            "what share",
        )
    )
    has_versioned_metric = any(
        term in normalized
        for term in (
            "dividend yield",
            "dividend",
            "strategy",
            "key strategy",
            "investment thesis",
            "noi margin",
            "occupancy",
            "capacity",
            "it capacity",
            "key facts",
            "quick facts",
            "fast facts",
            "headline stats",
            "key statistics",
            "portfolio",
            "cbd",
            "yield",
            "market share",
            "geographic mix",
        )
    )
    return has_simple_lookup and has_versioned_metric


def _is_visual_detail_lookup(normalized: str) -> bool:
    """Detect questions whose answer often lives in a chart/table/map/logos.

    These are not necessarily broad corpus questions; they are requests for ranked,
    spatial, logo-rendered, or named-list details that investor decks often encode as
    visuals. The label adds companion queries and lets retrieval favor structured chunks.
    """
    return any(
        term in normalized
        for term in (
            "ranked by",
            "ranked",
            "asset list",
            "property list",
            "named assets",
            "named properties",
            "property names",
            "which markets",
            "u.s. region",
            "us region",
            "regions vs",
            "region vs",
        )
    )


def _is_corpus_wide_scope(normalized: str) -> bool:
    """Detect questions requiring a scan across ALL documents to find the best/most/which entity.

    Only fires for open-ended "which company" and superlative questions where the answer
    cannot be found in a single or named-pair of documents.  Two-entity comparisons that
    already name both parties are handled by the comparative path instead.
    """
    # "which company", "which document", etc. — open-ended corpus scan
    has_which_who = any(
        term in normalized
        for term in (
            "which company",
            "which companies",
            "which document",
            "which documents",
            "which reit",
            "who has",
            "who appears",
        )
    )
    # Superlatives: "most directly positioned", "strongest case", etc.
    has_superlative = any(
        term in normalized
        for term in (
            "most directly",
            "most dependent",
            "most geographically",
            "most explicitly",
            "most global",
            "strongest case",
            "best positioned",
        )
    )
    # Corpus-scanning phrases: "across the sectors represented in these documents",
    # "across the uploaded materials", "in these documents", etc.
    has_corpus_scan_phrase = any(
        term in normalized
        for term in (
            "in these documents",
            "in the documents",
            "across the uploaded",
            "in the uploaded",
            "represented in these",
            "in the corpus",
            "the uploaded materials",
        )
    )
    return has_which_who or has_superlative or has_corpus_scan_phrase


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


def _corpus_wide_companions(question: str) -> list[str]:
    """Generate companion queries that spread retrieval across all documents in the corpus.

    These are generic paraphrases — not tied to specific company names — so they work
    for any financial document corpus. They help surface pages from documents that would
    otherwise be buried under a single dominant semantic match.
    """
    return [
        f"investor presentation overview highlights: {question}",
        f"financial metrics key statistics strategy: {question}",
    ]


def _visual_detail_companions(question: str) -> list[str]:
    return [
        f"{question} map figure table labels percentages",
        f"{question} ranked list portfolio breakdown",
    ]


def _contains_term(normalized: str, term: str) -> bool:
    """Match whole terms so 'annual' does not fire on 'annualized'."""
    if " " in term:
        return term in normalized
    return bool(re.search(rf"\b{re.escape(term)}\b", normalized))


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
