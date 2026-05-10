"""
Deterministic factuality scoring helpers for enterprise RAG answer evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from evaluation.common import safe_lower

VERDICT_CORRECT = "correct"
VERDICT_PARTIAL = "partially_correct"
VERDICT_INCORRECT = "incorrect"
VALID_VERDICTS = {VERDICT_CORRECT, VERDICT_PARTIAL, VERDICT_INCORRECT}
DEFAULT_VERDICT_SCORES = {
    VERDICT_CORRECT: 1.0,
    VERDICT_PARTIAL: 0.5,
    VERDICT_INCORRECT: 0.0,
}


@dataclass(slots=True)
class FactualityScoreRow:
    id: Any
    verdict: str
    score: float
    note: str
    category: str | None
    difficulty: str | None
    requires_multimodal: bool | None
    question: str | None
    answer: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "verdict": self.verdict,
            "score": self.score,
            "note": self.note,
            "category": self.category,
            "difficulty": self.difficulty,
            "requires_multimodal": self.requires_multimodal,
            "question": self.question,
            "answer": self.answer,
        }


@dataclass(slots=True)
class CategoryFactualitySummary:
    category: str
    total: int
    correct: int
    partially_correct: int
    incorrect: int
    weighted_factuality: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "total": self.total,
            "correct": self.correct,
            "partially_correct": self.partially_correct,
            "incorrect": self.incorrect,
            "weighted_factuality": self.weighted_factuality,
        }


@dataclass(slots=True)
class FactualityScoreSummary:
    total: int
    correct: int
    partially_correct: int
    incorrect: int
    weighted_factuality: float
    missing_answer_count: int
    categories: list[CategoryFactualitySummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "partially_correct": self.partially_correct,
            "incorrect": self.incorrect,
            "weighted_factuality": self.weighted_factuality,
            "missing_answer_count": self.missing_answer_count,
            "categories": [category.to_dict() for category in self.categories],
        }


@dataclass(slots=True)
class FactualityScoreResult:
    rows: list[FactualityScoreRow]
    summary: FactualityScoreSummary


def evaluate_factuality(
    *,
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
) -> FactualityScoreResult:
    questions_by_id = {row.get("id"): row for row in questions}
    answers_by_id = {row.get("id"): row for row in answers}
    gold_by_id = {row.get("id"): row for row in gold_rows}

    missing_gold_ids = [
        question_id for question_id in questions_by_id if question_id not in gold_by_id
    ]
    if missing_gold_ids:
        raise ValueError(
            f"Missing gold labels for question IDs: {missing_gold_ids[:10]}"
        )

    rows: list[FactualityScoreRow] = []
    for question in questions:
        question_id = question.get("id")
        if question_id is None:
            continue
        gold = gold_by_id[question_id]
        answer = answers_by_id.get(question_id)

        verdict = normalize_verdict(gold.get("verdict"))
        score = _resolve_score(verdict=verdict, value=gold.get("score"))
        note = str(gold.get("note") or "").strip()

        rows.append(
            FactualityScoreRow(
                id=question_id,
                verdict=verdict,
                score=score,
                note=note,
                category=question.get("category"),
                difficulty=question.get("difficulty"),
                requires_multimodal=question.get("requires_multimodal"),
                question=question.get("question"),
                answer=answer.get("answer") if answer else None,
            )
        )

    summary = _build_summary(rows)
    return FactualityScoreResult(rows=rows, summary=summary)


def normalize_verdict(value: Any) -> str:
    normalized = safe_lower(value)
    if normalized in {"partial", "partially correct", "partially_correct"}:
        return VERDICT_PARTIAL
    if normalized in {"incorrect", "wrong", "fail", "failed"}:
        return VERDICT_INCORRECT
    if normalized in {"correct", "pass", "passed"}:
        return VERDICT_CORRECT
    if normalized in VALID_VERDICTS:
        return normalized
    raise ValueError(f"Unsupported verdict value: {value!r}")


def _resolve_score(*, verdict: str, value: Any) -> float:
    if value is None or value == "":
        return DEFAULT_VERDICT_SCORES[verdict]
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid score value {value!r} for verdict {verdict}"
        ) from exc
    return max(0.0, min(1.0, score))


def _build_summary(rows: list[FactualityScoreRow]) -> FactualityScoreSummary:
    correct = sum(1 for row in rows if row.verdict == VERDICT_CORRECT)
    partial = sum(1 for row in rows if row.verdict == VERDICT_PARTIAL)
    incorrect = sum(1 for row in rows if row.verdict == VERDICT_INCORRECT)
    weighted = float(mean([row.score for row in rows])) if rows else 0.0
    missing_answers = sum(1 for row in rows if not row.answer)

    categories: list[CategoryFactualitySummary] = []
    by_category: dict[str, list[FactualityScoreRow]] = {}
    for row in rows:
        category = row.category or "uncategorized"
        by_category.setdefault(category, []).append(row)

    for category, items in sorted(by_category.items()):
        categories.append(
            CategoryFactualitySummary(
                category=category,
                total=len(items),
                correct=sum(1 for item in items if item.verdict == VERDICT_CORRECT),
                partially_correct=sum(
                    1 for item in items if item.verdict == VERDICT_PARTIAL
                ),
                incorrect=sum(1 for item in items if item.verdict == VERDICT_INCORRECT),
                weighted_factuality=(
                    float(mean([item.score for item in items])) if items else 0.0
                ),
            )
        )

    return FactualityScoreSummary(
        total=len(rows),
        correct=correct,
        partially_correct=partial,
        incorrect=incorrect,
        weighted_factuality=weighted,
        missing_answer_count=missing_answers,
        categories=categories,
    )
