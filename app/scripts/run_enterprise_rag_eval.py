"""
Run curated enterprise RAG evaluation questions and persist answers as JSONL.

Usage:
    uv run python -m app.scripts.run_enterprise_rag_eval
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any
import uuid

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.ai_provider import initialize_ai_provider
from app.core.logging_config import configure_logging
from app.db.schema import ensure_runtime_schema
from app.db.snowflake import SessionLocal, engine
from app.services.chat_conversation_service import ChatConversationService
from app.services.client_service import ClientLookupService

DEFAULT_CLIENT_NAME = "test_oai"
DEFAULT_REASONING_EFFORT = "high"
VALID_REASONING_EFFORTS = ("low", "medium", "high")
DEFAULT_INPUT_PATH = Path("enterprise_rag_eval_questions.jsonl")
DEFAULT_OUTPUT_PATH = Path("enterprise_rag_eval_answers_test4_oai.jsonl")
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF_SECONDS = 10.0
DEFAULT_WORKERS = 4
MAX_BACKOFF_SECONDS = 120.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalRunnerConfig:
    input_path: Path
    output_path: Path
    client_name: str
    reasoning_effort: str
    max_retries: int
    initial_backoff_seconds: float
    workers: int
    timestamped_output: bool
    debug_output_path: Path | None


@dataclass
class EvalRunStats:
    total_rows: int = 0
    processed_rows: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped_malformed: int = 0
    blank_rows: int = 0


@dataclass(frozen=True)
class WorkItem:
    line_no: int
    question_id: Any
    question: str


@dataclass
class QuestionExecutionResult:
    line_no: int
    question_id: Any
    question: str
    answer: str
    reasoning: str | None
    status: str
    attempts: int
    query_id: str | None
    latency_ms: int | None
    error: str | None


@dataclass(frozen=True)
class OutputPaths:
    output_path: Path
    output_tmp_path: Path
    debug_path: Path
    debug_tmp_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    client_id: str
    output_path: Path
    debug_path: Path
    manifest_path: Path
    started_at: datetime
    finished_at: datetime
    overwritten_output: bool


class EnterpriseRAGEvalRunner:
    """Executes a question set against a single client with retry semantics."""

    def __init__(self, config: EvalRunnerConfig):
        self.config = config

    def run(self) -> tuple[EvalRunStats, RunArtifacts]:
        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        self._bootstrap_runtime()
        client_id = self._resolve_client_id()

        stats = EvalRunStats()
        work_items = self._load_work_items(stats)
        if not work_items:
            raise ValueError(
                "No valid questions found in input file after parsing/skipping malformed rows."
            )

        logger.info(
            "Preflight complete: valid_questions=%d first_id=%s last_id=%s",
            len(work_items),
            work_items[0].question_id,
            work_items[-1].question_id,
        )

        output_paths = self._resolve_output_paths()
        overwritten_output = output_paths.output_path.exists()
        if overwritten_output:
            logger.warning(
                "Output file exists and will be replaced: %s", output_paths.output_path
            )

        results_by_line = self._run_parallel_queries(client_id, work_items, stats)
        self._validate_completeness(stats)

        self._write_outputs(output_paths, results_by_line)

        finished_at = datetime.now(timezone.utc)
        manifest = self._build_manifest(
            run_id=run_id,
            client_id=client_id,
            stats=stats,
            started_at=started_at,
            finished_at=finished_at,
            output_paths=output_paths,
        )
        output_paths.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        artifacts = RunArtifacts(
            run_id=run_id,
            client_id=client_id,
            output_path=output_paths.output_path,
            debug_path=output_paths.debug_path,
            manifest_path=output_paths.manifest_path,
            started_at=started_at,
            finished_at=finished_at,
            overwritten_output=overwritten_output,
        )
        return stats, artifacts

    def _bootstrap_runtime(self) -> None:
        ensure_runtime_schema(engine)
        initialize_ai_provider()

    def _resolve_client_id(self) -> str:
        with SessionLocal() as db:
            client = ClientLookupService(db).get_client_by_name(self.config.client_name)

        if client is None:
            raise ValueError(
                f"Client '{self.config.client_name}' not found in clients table."
            )
        return str(client.id)

    def _load_work_items(self, stats: EvalRunStats) -> list[WorkItem]:
        work_items: list[WorkItem] = []
        for line_no, raw_line in enumerate(
            self.config.input_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                stats.blank_rows += 1
                continue

            stats.total_rows += 1
            payload = self._parse_line(line, line_no, stats)
            if payload is None:
                continue

            work_items.append(
                WorkItem(
                    line_no=line_no,
                    question_id=payload["id"],
                    question=payload["question"],
                )
            )

        stats.processed_rows = len(work_items)
        return work_items

    def _parse_line(
        self, line: str, line_no: int, stats: EvalRunStats
    ) -> dict[str, Any] | None:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            stats.skipped_malformed += 1
            logger.warning("Skipping malformed JSONL line %d: %s", line_no, exc)
            return None

        question_id = row.get("id")
        question = row.get("question")
        if question_id is None or not isinstance(question, str) or not question.strip():
            stats.skipped_malformed += 1
            logger.warning(
                "Skipping line %d due to missing/invalid 'id' or 'question'.", line_no
            )
            return None

        return {"id": question_id, "question": question.strip()}

    def _run_parallel_queries(
        self,
        client_id: str,
        work_items: list[WorkItem],
        stats: EvalRunStats,
    ) -> dict[int, QuestionExecutionResult]:
        if not work_items:
            return {}

        max_workers = min(self.config.workers, len(work_items))
        results_by_line: dict[int, QuestionExecutionResult] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._execute_single_question, client_id, item): item
                for item in work_items
            }

            completed = 0
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = QuestionExecutionResult(
                        line_no=item.line_no,
                        question_id=item.question_id,
                        question=item.question,
                        answer=f"ERROR: {str(exc).strip() or repr(exc)}",
                        reasoning=None,
                        status="failed",
                        attempts=self.config.max_retries + 1,
                        query_id=None,
                        latency_ms=None,
                        error=str(exc).strip() or repr(exc),
                    )

                results_by_line[result.line_no] = result
                if result.status == "completed":
                    stats.succeeded += 1
                else:
                    stats.failed += 1

                completed += 1
                logger.info(
                    "Completed %d/%d questions (workers=%d).",
                    completed,
                    len(work_items),
                    max_workers,
                )

        return results_by_line

    def _execute_single_question(
        self,
        client_id: str,
        item: WorkItem,
    ) -> QuestionExecutionResult:
        total_attempts = self.config.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                with SessionLocal() as db:
                    result = ChatConversationService(db).execute_client_query(
                        client_id=client_id,
                        question=item.question,
                        reasoning_effort=self.config.reasoning_effort,
                    )

                answer = str(result.get("answer", "")).strip() or "No answer generated."
                reasoning_raw = result.get("reasoning")
                reasoning = (
                    str(reasoning_raw).strip() if reasoning_raw is not None else None
                )
                if reasoning == "":
                    reasoning = None

                return QuestionExecutionResult(
                    line_no=item.line_no,
                    question_id=item.question_id,
                    question=item.question,
                    answer=answer,
                    reasoning=reasoning,
                    status="completed",
                    attempts=attempt,
                    query_id=result.get("query_id"),
                    latency_ms=result.get("latency_ms"),
                    error=None,
                )
            except Exception as exc:
                message = str(exc).strip() or repr(exc)
                if attempt >= total_attempts:
                    logger.error(
                        "Question id=%s failed after %d attempts: %s",
                        item.question_id,
                        total_attempts,
                        message,
                    )
                    return QuestionExecutionResult(
                        line_no=item.line_no,
                        question_id=item.question_id,
                        question=item.question,
                        answer=f"ERROR: {message}",
                        reasoning=None,
                        status="failed",
                        attempts=attempt,
                        query_id=None,
                        latency_ms=None,
                        error=message,
                    )

                wait_seconds = min(
                    self.config.initial_backoff_seconds * (2 ** (attempt - 1)),
                    MAX_BACKOFF_SECONDS,
                )
                logger.warning(
                    "Question id=%s attempt %d/%d failed: %s. Retrying in %.1fs.",
                    item.question_id,
                    attempt,
                    total_attempts,
                    message,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

        return QuestionExecutionResult(
            line_no=item.line_no,
            question_id=item.question_id,
            question=item.question,
            answer="ERROR: unexpected retry flow",
            reasoning=None,
            status="failed",
            attempts=total_attempts,
            query_id=None,
            latency_ms=None,
            error="unexpected retry flow",
        )

    def _resolve_output_paths(self) -> OutputPaths:
        output_path = self._compute_final_output_path(self.config.output_path)
        debug_path = self._compute_debug_output_path(output_path)

        if output_path == debug_path:
            raise ValueError(
                "Main output path and debug output path cannot be the same."
            )

        manifest_path = self._derive_manifest_path(output_path)

        output_tmp = output_path.with_name(output_path.name + ".tmp")
        debug_tmp = debug_path.with_name(debug_path.name + ".tmp")

        return OutputPaths(
            output_path=output_path,
            output_tmp_path=output_tmp,
            debug_path=debug_path,
            debug_tmp_path=debug_tmp,
            manifest_path=manifest_path,
        )

    def _compute_final_output_path(self, base_output_path: Path) -> Path:
        if not self.config.timestamped_output:
            return base_output_path

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if base_output_path.suffix:
            filename = f"{base_output_path.stem}_{timestamp}{base_output_path.suffix}"
        else:
            filename = f"{base_output_path.name}_{timestamp}"
        return base_output_path.with_name(filename)

    def _compute_debug_output_path(self, output_path: Path) -> Path:
        if self.config.debug_output_path is not None:
            return self.config.debug_output_path

        suffix = output_path.suffix
        if suffix:
            return output_path.with_suffix(suffix + ".debug.jsonl")
        return output_path.with_name(output_path.name + ".debug.jsonl")

    def _derive_manifest_path(self, output_path: Path) -> Path:
        suffix = output_path.suffix
        if suffix:
            return output_path.with_suffix(suffix + ".manifest.json")
        return output_path.with_name(output_path.name + ".manifest.json")

    def _write_outputs(
        self,
        output_paths: OutputPaths,
        results_by_line: dict[int, QuestionExecutionResult],
    ) -> None:
        output_paths.output_path.parent.mkdir(parents=True, exist_ok=True)
        output_paths.debug_path.parent.mkdir(parents=True, exist_ok=True)

        self._safe_unlink(output_paths.output_tmp_path)
        self._safe_unlink(output_paths.debug_tmp_path)

        sorted_line_numbers = sorted(results_by_line.keys())

        with output_paths.output_tmp_path.open("w", encoding="utf-8") as output_fp:
            for line_no in sorted_line_numbers:
                result = results_by_line[line_no]
                output_row = {
                    "id": result.question_id,
                    "question": result.question,
                    "answer": result.answer,
                    "reasoning": result.reasoning,
                }
                output_fp.write(json.dumps(output_row, ensure_ascii=False) + "\n")

        with output_paths.debug_tmp_path.open("w", encoding="utf-8") as debug_fp:
            for line_no in sorted_line_numbers:
                result = results_by_line[line_no]
                debug_row = {
                    "id": result.question_id,
                    "line_no": result.line_no,
                    "status": result.status,
                    "attempts": result.attempts,
                    "query_id": result.query_id,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                }
                debug_fp.write(json.dumps(debug_row, ensure_ascii=False) + "\n")

        os.replace(output_paths.output_tmp_path, output_paths.output_path)
        os.replace(output_paths.debug_tmp_path, output_paths.debug_path)

    def _validate_completeness(self, stats: EvalRunStats) -> None:
        expected = stats.total_rows
        observed = stats.processed_rows + stats.skipped_malformed
        if observed != expected:
            raise RuntimeError(
                "Run completeness check failed: "
                f"processed({stats.processed_rows}) + skipped_malformed({stats.skipped_malformed}) "
                f"!= rows_read_non_blank({stats.total_rows})"
            )

    def _build_manifest(
        self,
        *,
        run_id: str,
        client_id: str,
        stats: EvalRunStats,
        started_at: datetime,
        finished_at: datetime,
        output_paths: OutputPaths,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "input_path": str(self.config.input_path),
            "input_non_blank_count": stats.total_rows,
            "client_name": self.config.client_name,
            "client_id": client_id,
            "reasoning_effort": self.config.reasoning_effort,
            "workers": self.config.workers,
            "max_retries": self.config.max_retries,
            "initial_backoff_seconds": self.config.initial_backoff_seconds,
            "timestamped_output": self.config.timestamped_output,
            "output_path": str(output_paths.output_path),
            "debug_output_path": str(output_paths.debug_path),
            "processed_rows": stats.processed_rows,
            "succeeded": stats.succeeded,
            "failed": stats.failed,
            "skipped_malformed": stats.skipped_malformed,
            "blank_rows": stats.blank_rows,
        }

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            logger.warning("Could not remove stale temp file: %s", path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run enterprise RAG eval questions and save answers as JSONL."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input questions JSONL file path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output answers JSONL file path (replaced atomically).",
    )
    parser.add_argument(
        "--client-name",
        default=DEFAULT_CLIENT_NAME,
        help="Client name to query.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        help=(
            f"Reasoning effort for generation ({'|'.join(VALID_REASONING_EFFORTS)})."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries per question after the initial attempt.",
    )
    parser.add_argument(
        "--initial-backoff-seconds",
        type=float,
        default=DEFAULT_INITIAL_BACKOFF_SECONDS,
        help="Initial retry backoff in seconds; doubles after each failed attempt.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel worker threads for question execution.",
    )
    parser.add_argument(
        "--timestamped-output",
        action="store_true",
        help="Append UTC timestamp to output filename for run history.",
    )
    parser.add_argument(
        "--debug-output",
        default=None,
        help=(
            "Optional debug JSONL output path for per-question metadata "
            "(id/query_id/status/attempts/error)."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> EvalRunnerConfig:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    client_name = str(args.client_name).strip()
    reasoning_effort = str(args.reasoning_effort).strip().lower()
    debug_output_path = (
        Path(args.debug_output).expanduser().resolve() if args.debug_output else None
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    if not client_name:
        raise ValueError("Client name cannot be empty.")
    if reasoning_effort not in VALID_REASONING_EFFORTS:
        allowed = ", ".join(VALID_REASONING_EFFORTS)
        raise ValueError(f"--reasoning-effort must be one of: {allowed}.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0.")
    if args.initial_backoff_seconds <= 0:
        raise ValueError("--initial-backoff-seconds must be > 0.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")

    return EvalRunnerConfig(
        input_path=input_path,
        output_path=output_path,
        client_name=client_name,
        reasoning_effort=reasoning_effort,
        max_retries=int(args.max_retries),
        initial_backoff_seconds=float(args.initial_backoff_seconds),
        workers=int(args.workers),
        timestamped_output=bool(args.timestamped_output),
        debug_output_path=debug_output_path,
    )


def main() -> int:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        config = _validate_args(args)
        runner = EnterpriseRAGEvalRunner(config)
        stats, artifacts = runner.run()
    except Exception as exc:
        logger.error("Eval run failed: %s", exc)
        print(f"[FAIL] {exc}")
        return 1

    print("Enterprise RAG eval run complete.")
    print(f"Run ID: {artifacts.run_id}")
    print(f"Input: {config.input_path}")
    print(f"Output: {artifacts.output_path}")
    print(f"Debug output: {artifacts.debug_path}")
    print(f"Manifest: {artifacts.manifest_path}")
    print(f"Client: {config.client_name} ({artifacts.client_id})")
    print(f"Reasoning effort: {config.reasoning_effort}")
    print(f"Workers: {config.workers}")
    print(f"Timestamped output: {config.timestamped_output}")
    print(f"Rows read (non-blank): {stats.total_rows}")
    print(f"Rows processed: {stats.processed_rows}")
    print(f"Succeeded: {stats.succeeded}")
    print(f"Failed: {stats.failed}")
    print(f"Skipped malformed: {stats.skipped_malformed}")
    print(f"Blank rows ignored: {stats.blank_rows}")
    if artifacts.overwritten_output:
        print("Warning: existing output file was replaced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
