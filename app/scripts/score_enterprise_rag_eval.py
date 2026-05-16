"""
Score enterprise RAG answers against a manually adjudicated gold JSONL file.

Usage:
    uv run python -m app.scripts.score_enterprise_rag_eval
"""

import argparse
import json
from pathlib import Path
from typing import Any

from app.evals import evaluate_factuality

DEFAULT_QUESTIONS_PATH = Path("enterprise_rag_eval_questions.jsonl")
DEFAULT_ANSWERS_PATH = Path("enterprise_rag_eval_answers_test4_oai.jsonl")
DEFAULT_GOLD_PATH = Path("enterprise_rag_eval_gold_test4.jsonl")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected object JSON at {path}:{line_no}")
        rows.append(parsed)
    return rows


def _derive_scored_output_path(answers_path: Path) -> Path:
    suffix = answers_path.suffix
    if suffix:
        return answers_path.with_suffix(f"{suffix}.scored.jsonl")
    return answers_path.with_name(f"{answers_path.name}.scored.jsonl")


def _derive_summary_output_path(scored_output_path: Path) -> Path:
    suffix = scored_output_path.suffix
    if suffix:
        return scored_output_path.with_suffix(f"{suffix}.summary.json")
    return scored_output_path.with_name(f"{scored_output_path.name}.summary.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score enterprise RAG answers against adjudicated gold labels."
    )
    parser.add_argument(
        "--questions",
        default=str(DEFAULT_QUESTIONS_PATH),
        help="Questions JSONL path.",
    )
    parser.add_argument(
        "--answers",
        default=str(DEFAULT_ANSWERS_PATH),
        help="Answers JSONL path.",
    )
    parser.add_argument(
        "--gold",
        default=str(DEFAULT_GOLD_PATH),
        help="Gold adjudication JSONL path (id/verdict/score/note).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Scored output JSONL path. Defaults to <answers>.scored.jsonl.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Summary JSON path. Defaults to <output>.summary.json.",
    )
    return parser


def _resolve_path(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def _validate_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    questions_path = _resolve_path(args.questions)
    answers_path = _resolve_path(args.answers)
    gold_path = _resolve_path(args.gold)
    output_path = (
        _resolve_path(args.output)
        if args.output
        else _derive_scored_output_path(answers_path)
    )
    summary_path = (
        _resolve_path(args.summary_output)
        if args.summary_output
        else _derive_summary_output_path(output_path)
    )

    _validate_file(questions_path, "Questions file")
    _validate_file(answers_path, "Answers file")
    _validate_file(gold_path, "Gold file")

    questions = _load_jsonl(questions_path)
    answers = _load_jsonl(answers_path)
    gold_rows = _load_jsonl(gold_path)

    result = evaluate_factuality(
        questions=questions,
        answers=answers,
        gold_rows=gold_rows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fp:
        for row in result.rows:
            fp.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    summary_payload = {
        "questions_path": str(questions_path),
        "answers_path": str(answers_path),
        "gold_path": str(gold_path),
        "output_path": str(output_path),
        "summary": result.summary.to_dict(),
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Scored rows: {len(result.rows)}")
    print(f"Weighted factuality: {result.summary.weighted_factuality:.4f}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
