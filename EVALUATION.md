# Testing and Evaluation Guide

This repo supports two verification modes:

- Automated tests (`pytest`) for code-level correctness.
- Enterprise RAG evaluations via `app/scripts/run_enterprise_rag_eval.py` for end-to-end answer quality checks.

## Prerequisites

From repo root (`/home/adiagarwal/vectera`):

```bash
cp .env.example .env
./setup.sh
docker-compose up -d qdrant
```

Notes:

- Runtime startup requires a valid AI API key (`AI_API_KEY` / `OPENAI_API_KEY` aliases).
- `setup_check` is stricter than runtime fallback and expects Snowflake + Qdrant + AI key.

## Run Tests

Run the full test suite:

```bash
uv run pytest
```

Run focused tests:

```bash
uv run pytest tests/test_retrieval.py
uv run pytest tests/test_retrieval.py::test_name
```

Run integration readiness check:

```bash
uv run python -m app.scripts.setup_check
```

## Run Enterprise RAG Evaluation

Script: `app/scripts/run_enterprise_rag_eval.py`

Basic usage (defaults):

```bash
uv run python -m app.scripts.run_enterprise_rag_eval
```

Default runtime values:

- `--input`: `enterprise_rag_eval_questions.jsonl`
- `--output`: `enterprise_rag_eval_answers_test4_oai.jsonl`
- `--client-name`: `test_oai`
- `--reasoning-effort`: `high` (`low|medium|high`)
- `--max-retries`: `5` (plus initial attempt)
- `--initial-backoff-seconds`: `10`
- `--workers`: `4`

### Input format

Input must be JSONL where each non-blank line contains:

```json
{"id": 1, "question": "What changed in Q4 revenue?"}
```

Rows with missing/invalid `id` or `question` are skipped and counted as malformed.

### Common command examples

Run with custom input/output:

```bash
uv run python -m app.scripts.run_enterprise_rag_eval \
  --input ./enterprise_rag_eval_questions.jsonl \
  --output ./artifacts/evals/answers.jsonl \
  --client-name test_oai
```

Enable timestamped output for run history:

```bash
uv run python -m app.scripts.run_enterprise_rag_eval \
  --output ./artifacts/evals/answers.jsonl \
  --timestamped-output
```

Tune retries/backoff/workers:

```bash
uv run python -m app.scripts.run_enterprise_rag_eval \
  --max-retries 3 \
  --initial-backoff-seconds 5 \
  --workers 8
```

Use a different reasoning effort:

```bash
uv run python -m app.scripts.run_enterprise_rag_eval \
  --reasoning-effort medium
```

Write debug output to an explicit path:

```bash
uv run python -m app.scripts.run_enterprise_rag_eval \
  --debug-output ./artifacts/evals/answers.debug.jsonl
```

## Output Files

Each run writes:

- Main output JSONL (`id`, `question`, `answer`, `reasoning`).
- Debug JSONL (`id`, `line_no`, `status`, `attempts`, `query_id`, `latency_ms`, `error`).
- Manifest JSON (`*.manifest.json`) with run metadata and stats.

Writes are atomic through temporary files and replace the destination on success.

## Troubleshooting

- `Client '<name>' not found`: ensure `--client-name` exists in the `clients` table.
- `Input file not found`: verify `--input` path from repo root or pass an absolute path.
- API key or provider initialization errors: verify `.env` has a valid key.
- Qdrant or DB connectivity errors: start dependencies (`docker-compose up -d qdrant`) and verify env settings.
