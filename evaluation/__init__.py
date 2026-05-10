"""
Evaluation helpers for retrieval and ingestion quality.
"""

from evaluation.factuality import (
    FactualityScoreResult,
    FactualityScoreRow,
    FactualityScoreSummary,
    evaluate_factuality,
)
from evaluation.ingestion import (
    IngestionEvalCase,
    IngestionEvalSummary,
    evaluate_ingestion_cases,
)
from evaluation.reporting import build_quality_payload
from evaluation.retrieval import (
    RetrievalEvalCase,
    RetrievalEvalSummary,
    evaluate_retrieval_cases,
)

__all__ = [
    "IngestionEvalCase",
    "IngestionEvalSummary",
    "FactualityScoreResult",
    "FactualityScoreRow",
    "FactualityScoreSummary",
    "RetrievalEvalCase",
    "RetrievalEvalSummary",
    "build_quality_payload",
    "evaluate_factuality",
    "evaluate_ingestion_cases",
    "evaluate_retrieval_cases",
]
