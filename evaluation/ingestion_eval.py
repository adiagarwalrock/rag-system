"""
Ingestion evaluation helpers.

These evaluate parse/indexing runs from recorded statuses and metadata counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from evaluation.common import mean_or_zero, safe_lower


@dataclass(slots=True)
class IngestionEvalCase:
    document_id: str
    parse_success: bool
    chunk_count: int
    vector_node_count: int
    required_metadata_fields: list[str]
    metadata_by_chunk: list[dict[str, Any]] = field(default_factory=list)
    indexing_success: bool = True
    error_message: str | None = None


@dataclass(slots=True)
class IngestionCaseScore:
    document_id: str
    parse_success: float
    chunk_count: int
    vector_node_count: int
    metadata_completeness: float
    qdrant_index_success: float


@dataclass(slots=True)
class IngestionEvalSummary:
    case_count: int
    parse_success_rate: float
    avg_chunk_count: float
    avg_vector_node_count: float
    avg_metadata_completeness: float
    qdrant_index_success_rate: float
    cases: list[IngestionCaseScore]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "parse_success_rate": self.parse_success_rate,
            "avg_chunk_count": self.avg_chunk_count,
            "avg_vector_node_count": self.avg_vector_node_count,
            "avg_metadata_completeness": self.avg_metadata_completeness,
            "qdrant_index_success_rate": self.qdrant_index_success_rate,
            "cases": [
                {
                    "document_id": case.document_id,
                    "parse_success": case.parse_success,
                    "chunk_count": case.chunk_count,
                    "vector_node_count": case.vector_node_count,
                    "metadata_completeness": case.metadata_completeness,
                    "qdrant_index_success": case.qdrant_index_success,
                }
                for case in self.cases
            ],
        }


def evaluate_ingestion_cases(cases: list[IngestionEvalCase]) -> IngestionEvalSummary:
    scored = [_score_case(case) for case in cases]
    return IngestionEvalSummary(
        case_count=len(scored),
        parse_success_rate=mean_or_zero([case.parse_success for case in scored]),
        avg_chunk_count=mean_or_zero([case.chunk_count for case in scored]),
        avg_vector_node_count=mean_or_zero([case.vector_node_count for case in scored]),
        avg_metadata_completeness=mean_or_zero(
            [case.metadata_completeness for case in scored]
        ),
        qdrant_index_success_rate=mean_or_zero(
            [case.qdrant_index_success for case in scored]
        ),
        cases=scored,
    )


def _score_case(case: IngestionEvalCase) -> IngestionCaseScore:
    metadata_completeness = _metadata_completeness(
        case.metadata_by_chunk, case.required_metadata_fields
    )
    return IngestionCaseScore(
        document_id=case.document_id,
        parse_success=1.0 if case.parse_success else 0.0,
        chunk_count=case.chunk_count,
        vector_node_count=case.vector_node_count,
        metadata_completeness=metadata_completeness,
        qdrant_index_success=1.0 if case.indexing_success else 0.0,
    )


def _metadata_completeness(
    metadata_by_chunk: list[dict[str, Any]], required_fields: list[str]
) -> float:
    if not metadata_by_chunk or not required_fields:
        return 0.0

    per_chunk_scores: list[float] = []
    required = [
        required_field
        for required_field in required_fields
        if safe_lower(required_field)
    ]
    if not required:
        return 0.0

    for metadata in metadata_by_chunk:
        present = 0
        for required_field in required:
            value = metadata.get(required_field)
            if value is not None and value != "":
                present += 1
        per_chunk_scores.append(present / len(required))

    return float(mean(per_chunk_scores)) if per_chunk_scores else 0.0
