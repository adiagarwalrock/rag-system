import json
from pathlib import Path

from app.scripts.run_enterprise_rag_eval import (
    EnterpriseRAGEvalRunner,
    EvalRunnerConfig,
    OutputPaths,
    QuestionExecutionResult,
)


def test_eval_debug_output_includes_retrieval_diagnostics_without_changing_answers(
    tmp_path,
):
    output_path = tmp_path / "answers.jsonl"
    debug_path = tmp_path / "answers.jsonl.debug.jsonl"
    paths = OutputPaths(
        output_path=output_path,
        output_tmp_path=Path(str(output_path) + ".tmp"),
        debug_path=debug_path,
        debug_tmp_path=Path(str(debug_path) + ".tmp"),
        manifest_path=tmp_path / "answers.jsonl.manifest.json",
    )
    config = EvalRunnerConfig(
        input_path=tmp_path / "questions.jsonl",
        output_path=output_path,
        client_name="client",
        reasoning_effort="high",
        max_retries=0,
        initial_backoff_seconds=1.0,
        workers=1,
        timestamped_output=False,
        debug_output_path=None,
        question_timeout_seconds=360,
    )
    result = QuestionExecutionResult(
        line_no=1,
        question_id=1,
        question="What changed?",
        answer="Final answer [1].",
        reasoning="Trace",
        status="completed",
        attempts=1,
        query_id="query-1",
        latency_ms=123,
        error=None,
        diagnostics={
            "evidence_count": 2,
            "source_count": 10,
            "retrieval_mode": "hybrid",
            "query_expanded": True,
            "retrieval_strategy": "decomposed",
            "router_reason": "Temporal comparison.",
            "routed_queries": ["older plan", "newer update"],
            "router_fallback_reason": None,
            "retrieval_diagnostics": {"evidence_document_count": 2},
            "intent_labels": ["temporal_delta"],
            "companion_queries": ["older plan", "newer update"],
            "companion_counts_by_query": {"older plan": 3},
            "evidence_by_document": {"Doc": 2},
            "evidence_by_version_group": {"v1": 1, "v2": 1},
            "evidence_by_entity": {"BXP": 2},
            "top_citations": [{"citation_label": "Doc p.1"}],
        },
    )

    EnterpriseRAGEvalRunner(config)._write_outputs(paths, {1: result})

    answer_row = json.loads(output_path.read_text(encoding="utf-8"))
    debug_row = json.loads(debug_path.read_text(encoding="utf-8"))

    assert answer_row == {
        "id": 1,
        "question": "What changed?",
        "answer": "Final answer [1].",
        "reasoning": "Trace",
    }
    assert debug_row["evidence_count"] == 2
    assert debug_row["source_count"] == 10
    assert debug_row["retrieval_mode"] == "hybrid"
    assert debug_row["query_expanded"] is True
    assert debug_row["retrieval_strategy"] == "decomposed"
    assert debug_row["router_reason"] == "Temporal comparison."
    assert debug_row["routed_queries"] == ["older plan", "newer update"]
    assert debug_row["intent_labels"] == ["temporal_delta"]
    assert debug_row["companion_queries"] == ["older plan", "newer update"]
    assert debug_row["companion_counts_by_query"] == {"older plan": 3}
    assert debug_row["evidence_by_document"] == {"Doc": 2}
    assert debug_row["evidence_by_version_group"] == {"v1": 1, "v2": 1}
    assert debug_row["evidence_by_entity"] == {"BXP": 2}
    assert debug_row["top_citations"] == [{"citation_label": "Doc p.1"}]
