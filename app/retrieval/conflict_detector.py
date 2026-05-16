"""
Conflict detector: identifies conflicting facts across retrieved chunks
from different document versions or sources.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CONFLICT_QUERY_TERMS = (
    "conflict",
    "conflicting",
    "contradict",
    "contradiction",
    "across documents",
    "across document",
    "across sources",
    "between documents",
    "disagree",
)

NUMERIC_VALUE_PATTERN = re.compile(
    r"(?<!\w)(?:[$€£])?\d{1,4}(?:,\d{3})*(?:\.\d+)?"
    r"(?:\s*(?:%|percent|bps|basis points|million|billion|thousand|bn|mm|m|k|x))?(?!\w)",
    re.IGNORECASE,
)

COMMON_TOPIC_STOPWORDS = {
    "the",
    "this",
    "that",
    "with",
    "from",
    "into",
    "during",
    "were",
    "was",
    "and",
    "for",
    "is",
    "are",
    "by",
    "of",
    "to",
    "in",
}

CONTEXT_STOPWORDS = {
    *COMMON_TOPIC_STOPWORDS,
    "data",
    "year",
    "years",
    "period",
    "ended",
}


@dataclass(slots=True, frozen=True)
class NumericFact:
    context_label: str
    context_tokens: frozenset[str]
    value_raw: str
    value: float | None
    unit: str


@dataclass(slots=True)
class NodeNumericFacts:
    node: Any
    document_name: str
    source_name_key: str
    document_id: str | None
    version_label: str
    version_group: str
    facts: list[NumericFact]


def detect_conflicts(
    source_nodes: list,
    evidence_nodes: list | None = None,
    max_conflicts: int = 3,
    question: str | None = None,
) -> list[dict[str, Any]]:
    """
    Analyze retrieved nodes for high-confidence conflicting information.

    Detects:
    - Numeric disagreements on similar topics across version-related sources.

    Args:
        source_nodes: List of NodeWithScore objects
        evidence_nodes: Evidence selected for synthesis (plus near-miss candidates)
        max_conflicts: Maximum conflicts to return
        question: Optional question text, used to decide when cross-document
            conflict checks should run.

    Returns:
        List of conflict dicts with type, summary, and supporting chunk info.
    """
    if not source_nodes:
        return []

    candidates = _select_conflict_candidates(
        source_nodes, evidence_nodes or [], question or ""
    )
    if len(candidates) < 2:
        return []

    conflicts: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    node_facts = _collect_node_facts(candidates)
    if len(node_facts) < 2:
        return []

    # Group by version family; conflicts across unrelated families are noisy.
    within_group_conflicts = _find_numeric_conflicts(
        node_facts=node_facts,
        seen_signatures=seen_signatures,
        allow_cross_group=False,
        min_topic_overlap=2,
        min_context_overlap=2,
        max_conflicts=max_conflicts,
    )
    conflicts.extend(within_group_conflicts)

    if len(conflicts) < max_conflicts and _should_check_cross_document(question or ""):
        cross_group_conflicts = _find_numeric_conflicts(
            node_facts=node_facts,
            seen_signatures=seen_signatures,
            allow_cross_group=True,
            min_topic_overlap=3,
            min_context_overlap=2,
            max_conflicts=max_conflicts - len(conflicts),
        )
        conflicts.extend(cross_group_conflicts)

    logger.info(
        "Detected %d conflicts across %d candidate nodes",
        len(conflicts),
        len(candidates),
    )
    return conflicts


def _collect_node_facts(nodes: list) -> list[NodeNumericFacts]:
    collected: list[NodeNumericFacts] = []
    for node in nodes:
        metadata = node.node.metadata or {}
        facts = _extract_numeric_facts(node.node.text or "")
        if not facts:
            continue
        collected.append(
            NodeNumericFacts(
                node=node,
                facts=facts,
                document_name=metadata.get("document_name")
                or metadata.get("source_file")
                or "Unknown",
                source_name_key=_normalized_source_name(
                    metadata.get("document_name")
                    or metadata.get("source_file")
                    or "unknown"
                ),
                document_id=metadata.get("document_id"),
                version_label=(metadata.get("version_label") or "unknown").strip()
                or "unknown",
                version_group=str(
                    metadata.get("document_version_group")
                    or metadata.get("document_family")
                    or metadata.get("document_id")
                    or "unknown"
                ),
            )
        )
    return collected


def _find_numeric_conflicts(
    node_facts: list[NodeNumericFacts],
    seen_signatures: set[str],
    allow_cross_group: bool,
    min_topic_overlap: int,
    min_context_overlap: int,
    max_conflicts: int,
) -> list[dict[str, Any]]:
    from itertools import combinations

    conflicts: list[dict[str, Any]] = []
    for left, right in combinations(node_facts, 2):
        if _is_non_conflict_pair(left, right, allow_cross_group):
            continue
        if not _shares_topic(
            left.node.node.text or "",
            right.node.node.text or "",
            min_shared=min_topic_overlap,
        ):
            continue

        for left_fact in left.facts:
            for right_fact in right.facts:
                if not _facts_conflict(
                    left_fact, right_fact, min_context_overlap=min_context_overlap
                ):
                    continue

                signature = _conflict_signature(left, right, left_fact, right_fact)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)

                overlap_tokens = sorted(
                    left_fact.context_tokens & right_fact.context_tokens
                )
                overlap_label = (
                    " ".join(overlap_tokens[:4])
                    if overlap_tokens
                    else left_fact.context_label
                )
                conflicts.append(
                    {
                        "conflict_type": "numeric_disagreement",
                        "summary": (
                            f"Conflicting values for '{overlap_label}': "
                            f"'{left_fact.value_raw}' in {left.document_name} ({left.version_label}) "
                            f"vs '{right_fact.value_raw}' in {right.document_name} ({right.version_label})"
                        ),
                        "supporting_chunks": [
                            {
                                "document_name": left.document_name,
                                "version_label": left.version_label,
                                "text_snippet": (left.node.node.text or "")[:220],
                            },
                            {
                                "document_name": right.document_name,
                                "version_label": right.version_label,
                                "text_snippet": (right.node.node.text or "")[:220],
                            },
                        ],
                    }
                )
                if len(conflicts) >= max_conflicts:
                    return conflicts
    return conflicts


def _is_non_conflict_pair(
    left: NodeNumericFacts,
    right: NodeNumericFacts,
    allow_cross_group: bool,
) -> bool:
    # Same document + same version disagreements are usually chunking noise.
    if left.document_id and left.document_id == right.document_id:
        if left.version_label == right.version_label:
            return True
    if left.source_name_key and left.source_name_key == right.source_name_key:
        if left.version_label == right.version_label:
            return True
        if "unknown" in {left.version_label, right.version_label}:
            return True
    if not allow_cross_group and left.version_group != right.version_group:
        return True
    return False


def _facts_conflict(
    left: NumericFact,
    right: NumericFact,
    min_context_overlap: int,
) -> bool:
    if left.value_raw == right.value_raw:
        return False

    if left.unit != right.unit:
        # Keep comparisons unit-consistent to reduce false positives.
        return False

    context_overlap = left.context_tokens & right.context_tokens
    if len(context_overlap) < min_context_overlap:
        return False

    if left.value is None or right.value is None:
        return True

    return abs(left.value - right.value) > _numeric_tolerance(left.unit)


def _numeric_tolerance(unit: str) -> float:
    if unit == "percent":
        return 0.1
    if unit == "basis_points":
        return 1.0
    return 0.01


def _conflict_signature(
    left_node: NodeNumericFacts,
    right_node: NodeNumericFacts,
    left_fact: NumericFact,
    right_fact: NumericFact,
) -> str:
    participants = sorted(
        [
            str(left_node.document_id or left_node.source_name_key),
            str(right_node.document_id or right_node.source_name_key),
        ]
    )
    context = "|".join(sorted(left_fact.context_tokens & right_fact.context_tokens))
    values = sorted([left_fact.value_raw, right_fact.value_raw])
    return f"{context}|{participants[0]}|{participants[1]}|{values[0]}|{values[1]}"


def _extract_numeric_facts(text: str) -> list[NumericFact]:
    """
    Extract numeric facts with lightweight context windows.

    We keep this parser intentionally deterministic and conservative:
    - derive a local token context around each numeric mention
    - skip likely chart-axis lines with many numbers but little prose
    - keep unit-aware numeric normalization for better comparisons
    """
    facts: list[NumericFact] = []
    if not text:
        return facts

    segments = re.split(r"[\n\.]+", text)
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        if _is_probably_axis_or_table_line(segment):
            continue

        for match in NUMERIC_VALUE_PATTERN.finditer(segment):
            raw_value = match.group(0).strip()
            value, unit = _parse_numeric_value(raw_value)
            if value is None:
                continue
            if unit == "count" and _is_probable_year(value):
                continue
            if unit == "count" and value.is_integer() and value <= 5:
                # Usually list/axis ordinals (1, 2, 3...) rather than factual claims.
                continue

            context = _context_tokens_around(segment, match.start(), match.end())
            if len(context) < 2:
                continue

            context_tokens = frozenset(context)
            facts.append(
                NumericFact(
                    context_label=" ".join(context[:4]),
                    context_tokens=context_tokens,
                    value_raw=raw_value,
                    value=value,
                    unit=unit,
                )
            )
    return facts


def _is_probably_axis_or_table_line(text: str) -> bool:
    numbers = NUMERIC_VALUE_PATTERN.findall(text)
    words = re.findall(r"[A-Za-z]{3,}", text)
    # Keep mixed prose+numeric lines (often the only source for chart values).
    return len(numbers) >= 14 and len(words) <= 8


def _context_tokens_around(text: str, start: int, end: int) -> list[str]:
    left = text[max(0, start - 70) : start]
    right = text[end : min(len(text), end + 70)]
    window = f"{left} {right}".strip()
    tokens = re.findall(r"[a-zA-Z]{3,}", window.lower())
    return _dedupe_filtered_tokens(tokens, CONTEXT_STOPWORDS, max_items=7)


def _parse_numeric_value(raw_value: str) -> tuple[float | None, str]:
    lowered = raw_value.lower().strip()
    value_match = re.search(r"[-+]?\d+(?:\.\d+)?", lowered.replace(",", ""))
    if not value_match:
        return None, "count"

    try:
        value = float(value_match.group(0))
    except ValueError:
        return None, "count"

    unit = "count"
    if "%" in lowered or "percent" in lowered:
        unit = "percent"
    elif "basis points" in lowered or "bps" in lowered:
        unit = "basis_points"
    elif any(sym in lowered for sym in ("$", "€", "£")):
        unit = "currency"
    elif lowered.endswith("bn") or "billion" in lowered:
        value *= 1_000_000_000
    elif lowered.endswith("mm") or "million" in lowered:
        value *= 1_000_000
    elif lowered.endswith("k") or "thousand" in lowered:
        value *= 1_000

    return value, unit


def _is_probable_year(value: float) -> bool:
    return value.is_integer() and 1900 <= int(value) <= 2100


def _shares_topic(text_a: str, text_b: str, min_shared: int = 2) -> bool:
    tokens_a = _topic_tokens(text_a)
    tokens_b = _topic_tokens(text_b)
    return len(tokens_a & tokens_b) >= min_shared


def _topic_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
    return set(_dedupe_filtered_tokens(tokens, COMMON_TOPIC_STOPWORDS))


def _dedupe_filtered_tokens(
    tokens: list[str],
    stopwords: set[str],
    max_items: int | None = None,
) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in stopwords or token in seen:
            continue
        seen.add(token)
        filtered.append(token)
        if max_items is not None and len(filtered) >= max_items:
            break
    return filtered


def _node_key(node: Any) -> str:
    metadata = node.node.metadata or {}
    return str(
        getattr(node.node, "node_id", None)
        or metadata.get("chunk_id")
        or metadata.get("document_id")
        or (node.node.text or "")[:80]
    )


def _select_conflict_candidates(
    source_nodes: list, evidence_nodes: list, question: str
) -> list:
    selected = []
    seen: set[str] = set()

    for node in evidence_nodes:
        key = _node_key(node)
        if key in seen:
            continue
        seen.add(key)
        selected.append(node)

    near_miss_budget = 2 if evidence_nodes else 4
    if _should_check_cross_document(question):
        near_miss_budget = max(near_miss_budget, 5)

    for node in source_nodes:
        if near_miss_budget <= 0:
            break
        key = _node_key(node)
        if key in seen:
            continue
        seen.add(key)
        selected.append(node)
        near_miss_budget -= 1

    if selected:
        return selected
    return source_nodes[:6]


def _should_check_cross_document(question: str) -> bool:
    normalized = (question or "").lower()
    return any(term in normalized for term in CONFLICT_QUERY_TERMS)


def _normalized_source_name(raw_name: str) -> str:
    normalized = (raw_name or "").strip().lower()
    return re.sub(r"\s+", " ", normalized)
