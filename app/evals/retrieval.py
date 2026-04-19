"""
Retrieval evaluation helpers.

These functions score retrieval cases from local result dictionaries or
captured query logs without depending on Qdrant, Snowflake, or Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.evals.common import mean_or_zero, safe_lower, unique_nonempty


@dataclass(slots=True)
class RetrievalEvalCase:
    query: str
    relevant_ids: list[str]
    retrieved: list[dict[str, Any]]
    expected_version_label: str | None = None
    expect_conflict: bool = False
    expected_citation_labels: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalCaseScore:
    query: str
    recall_at_5: float
    recall_at_10: float
    hit_rate_at_5: float
    mrr: float
    citation_precision: float
    version_attribution: float
    conflict_attribution: float
    candidate_count: int


@dataclass(slots=True)
class RetrievalEvalSummary:
    case_count: int
    mean_recall_at_5: float
    mean_recall_at_10: float
    mean_hit_rate_at_5: float
    mean_mrr: float
    mean_citation_precision: float
    mean_version_attribution: float
    mean_conflict_attribution: float
    cases: list[RetrievalCaseScore]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "mean_recall_at_5": self.mean_recall_at_5,
            "mean_recall_at_10": self.mean_recall_at_10,
            "mean_hit_rate_at_5": self.mean_hit_rate_at_5,
            "mean_mrr": self.mean_mrr,
            "mean_citation_precision": self.mean_citation_precision,
            "mean_version_attribution": self.mean_version_attribution,
            "mean_conflict_attribution": self.mean_conflict_attribution,
            "cases": [
                {
                    "query": case.query,
                    "recall_at_5": case.recall_at_5,
                    "recall_at_10": case.recall_at_10,
                    "hit_rate_at_5": case.hit_rate_at_5,
                    "mrr": case.mrr,
                    "citation_precision": case.citation_precision,
                    "version_attribution": case.version_attribution,
                    "conflict_attribution": case.conflict_attribution,
                    "candidate_count": case.candidate_count,
                }
                for case in self.cases
            ],
        }


def evaluate_retrieval_cases(cases: list[RetrievalEvalCase]) -> RetrievalEvalSummary:
    scored = [_score_case(case) for case in cases]
    return RetrievalEvalSummary(
        case_count=len(scored),
        mean_recall_at_5=mean_or_zero([case.recall_at_5 for case in scored]),
        mean_recall_at_10=mean_or_zero([case.recall_at_10 for case in scored]),
        mean_hit_rate_at_5=mean_or_zero([case.hit_rate_at_5 for case in scored]),
        mean_mrr=mean_or_zero([case.mrr for case in scored]),
        mean_citation_precision=mean_or_zero(
            [case.citation_precision for case in scored]
        ),
        mean_version_attribution=mean_or_zero(
            [case.version_attribution for case in scored]
        ),
        mean_conflict_attribution=mean_or_zero(
            [case.conflict_attribution for case in scored]
        ),
        cases=scored,
    )


def _score_case(case: RetrievalEvalCase) -> RetrievalCaseScore:
    ranked_ids = _ranked_ids(case.retrieved)
    gold = {safe_lower(item) for item in case.relevant_ids if safe_lower(item)}

    recall_at_5 = _recall_at_k(ranked_ids, gold, 5)
    recall_at_10 = _recall_at_k(ranked_ids, gold, 10)
    hit_rate_at_5 = 1.0 if _first_hit(ranked_ids, gold, 5) else 0.0
    mrr = _mrr(ranked_ids, gold)
    citation_precision = _citation_precision(
        case.retrieved, case.expected_citation_labels
    )
    version_attribution = _version_attribution(
        case.retrieved, case.expected_version_label
    )
    conflict_attribution = _conflict_attribution(case.retrieved, case.expect_conflict)

    return RetrievalCaseScore(
        query=case.query,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        hit_rate_at_5=hit_rate_at_5,
        mrr=mrr,
        citation_precision=citation_precision,
        version_attribution=version_attribution,
        conflict_attribution=conflict_attribution,
        candidate_count=len(ranked_ids),
    )


def _ranked_ids(retrieved: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for item in retrieved:
        candidate = (
            item.get("vector_node_id")
            or item.get("node_id")
            or item.get("chunk_id")
            or item.get("document_id")
        )
        if candidate:
            ids.append(str(candidate))
    return unique_nonempty(ids)


def _recall_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top_k = ranked_ids[:k]
    return len({safe_lower(item) for item in top_k} & gold) / len(gold)


def _first_hit(ranked_ids: list[str], gold: set[str], k: int) -> bool:
    return any(safe_lower(item) in gold for item in ranked_ids[:k])


def _mrr(ranked_ids: list[str], gold: set[str]) -> float:
    for index, candidate in enumerate(ranked_ids, start=1):
        if safe_lower(candidate) in gold:
            return 1.0 / index
    return 0.0


def _citation_precision(
    retrieved: list[dict[str, Any]], expected_citation_labels: list[str]
) -> float:
    expected = {
        safe_lower(label) for label in expected_citation_labels if safe_lower(label)
    }
    if not expected:
        return 0.0

    cited_labels = []
    for item in retrieved:
        candidate = item.get("citation_label") or item.get("document_name")
        normalized = safe_lower(candidate)
        if normalized:
            cited_labels.append(normalized)

    if not cited_labels:
        return 0.0

    hits = sum(1 for label in cited_labels if label in expected)
    return hits / len(cited_labels)


def _version_attribution(
    retrieved: list[dict[str, Any]], expected_version_label: str | None
) -> float:
    if not expected_version_label:
        return 0.0
    expected = safe_lower(expected_version_label)
    for item in retrieved:
        version = safe_lower(item.get("version_label"))
        if version == expected:
            return 1.0
    return 0.0


def _conflict_attribution(
    retrieved: list[dict[str, Any]], expect_conflict: bool
) -> float:
    if not expect_conflict:
        return 1.0
    for item in retrieved:
        if item.get("conflict_type") or item.get("conflicts"):
            return 1.0
    return 0.0

