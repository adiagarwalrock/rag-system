from __future__ import annotations

from typing import Any


def synthesize_answer(
    retriever: Any,
    question: str,
    citations: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> dict[str, Any]:
    return retriever.answer_synthesizer.synthesize(
        question=question,
        citations=citations,
        conflicts=conflicts,
    )
