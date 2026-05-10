from evaluation.factuality import (
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_PARTIAL,
    evaluate_factuality,
    normalize_verdict,
)


def test_normalize_verdict_accepts_expected_aliases():
    assert normalize_verdict("correct") == VERDICT_CORRECT
    assert normalize_verdict("partially correct") == VERDICT_PARTIAL
    assert normalize_verdict("partial") == VERDICT_PARTIAL
    assert normalize_verdict("incorrect") == VERDICT_INCORRECT


def test_evaluate_factuality_builds_weighted_summary_and_category_rollup():
    questions = [
        {
            "id": 1,
            "category": "tables",
            "difficulty": "easy",
            "requires_multimodal": False,
            "question": "Q1",
        },
        {
            "id": 2,
            "category": "tables",
            "difficulty": "medium",
            "requires_multimodal": True,
            "question": "Q2",
        },
        {
            "id": 3,
            "category": "charts",
            "difficulty": "hard",
            "requires_multimodal": True,
            "question": "Q3",
        },
    ]
    answers = [
        {"id": 1, "answer": "A1"},
        {"id": 2, "answer": "A2"},
        {"id": 3, "answer": "A3"},
    ]
    gold_rows = [
        {"id": 1, "verdict": "correct", "score": 1.0, "note": "ok"},
        {"id": 2, "verdict": "partially_correct", "score": 0.5, "note": "partial"},
        {"id": 3, "verdict": "incorrect", "score": 0.0, "note": "wrong"},
    ]

    result = evaluate_factuality(
        questions=questions,
        answers=answers,
        gold_rows=gold_rows,
    )

    assert len(result.rows) == 3
    assert result.summary.total == 3
    assert result.summary.correct == 1
    assert result.summary.partially_correct == 1
    assert result.summary.incorrect == 1
    assert result.summary.weighted_factuality == 0.5

    by_category = {row.category: row for row in result.summary.categories}
    assert by_category["tables"].total == 2
    assert by_category["tables"].weighted_factuality == 0.75
    assert by_category["charts"].total == 1
    assert by_category["charts"].weighted_factuality == 0.0
