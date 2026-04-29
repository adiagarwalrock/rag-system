# AGENTS.md

## Quick Start (repo root)

- `cp .env.example .env && ./setup.sh` (`setup.sh` requires `uv` + `npm`, installs Python deps, and installs global `@llamaindex/liteparse`).
- Start Qdrant before ingestion/retrieval/chat: `docker-compose up -d qdrant`.
- Streamlit primary app: `uv run streamlit run streamlit_app.py --server.port 8502`.
- Optional REST app: `uv run uvicorn api:app --reload --port 8000`.

## Runtime Wiring

- `streamlit_app.py` and `api.py` share the same service layer; Streamlit uses `ui/lib/api.py` (`VecteraCore`) and does not call local HTTP routes.
- Keep business logic in `app/services/*`; keep API routes (`app/api/*`) and UI pages (`ui/pages/*`) thin.
- Boundaries: orchestration `app/services/`, ingestion `app/ingestion/`, retrieval `app/retrieval/`, vector infra `app/indexing/`, DB/session state `app/db/`.

## Env + Integration Gotchas

- Settings load from `.env` via `pydantic-settings` (`app/core/config.py`); `AI_API_KEY` accepts aliases (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`).
- Startup is strict: `validate_runtime_settings()` fails on missing/placeholder API key (no mock fallback).
- If `SNOWFLAKE_ACCOUNT` and `SNOWFLAKE_USER` are unset, runtime falls back to SQLite `rag_local.db` (`app/db/snowflake.py`).
- Schema is migration-less and ensured at runtime via `ensure_runtime_schema(...)` (`api.py`, `ui/lib/api.py`); keep schema changes additive-safe.
- Qdrant vector dimensions are enforced; changing `VECTOR_DIMENSIONS`/`EMBEDDING_OUTPUT_DIMENSION` on an existing collection may require collection recreation.

## Verification

- Full test suite: `uv run pytest`.
- Focused tests: `uv run pytest tests/test_retrieval.py` or `uv run pytest tests/test_retrieval.py::test_name`.
- Integration readiness check: `uv run python -m app.scripts.setup_check`.
- `setup_check` is stricter than runtime fallback: Snowflake + Qdrant + valid AI key must all pass.
- Tests use in-memory SQLite fixtures (`tests/conftest.py`), so most tests do not require Snowflake/Qdrant.

## Agent Persona

- Act as a senior staff FAANG engineer for this repo: design for end-to-end system coherence, prefer sleek low-overhead performance, use abstractions only when they materially help, and keep solutions portable across local, Docker, and CI.

## Engineering Rules For Agents

- Treat the repo as one integrated system, not isolated files: optimize for end-to-end behavior and consistency across `app/`, `ui/`, API, retrieval, ingestion, and DB layers.
- When changing one module, verify dependent contracts/callers (schemas, service interfaces, metadata expectations, runtime wiring) so the whole codebase stays in sync.
- Engineer changes at a senior staff bar: production-grade design, clear ownership boundaries, measurable performance impact, and maintainability under long-term iteration.
- Prefer sleek, low-overhead code paths: avoid unnecessary allocations/indirection, keep hot paths simple, and choose the most efficient approach that preserves readability.
- Use higher-order/composable functions deliberately where they improve reuse and clarity, but avoid functional abstraction layers that add runtime cost without clear value.
- Keep implementations portable across local dev, Docker, and CI: avoid platform-specific assumptions and favor standard-library/Python-native patterns unless a dependency is justified.
- Prefer OOP for new core flows: add cohesive classes/managers/services instead of procedural sprawl.
- Cache expensive/shared objects instead of recreating per call:
  - LLM/embeddings: `app/core/ai_provider.py` (`initialize_ai_provider`, `get_llm`, `get_embeddings`).
  - DB connection/session factory: `app/db/snowflake.py` module globals (`engine`, `SessionLocal`).
  - Qdrant/vector and ingestion runtime objects: `vector_store_manager`, `chat_history_store`, ingestion queue manager.
- Place utilities/helpers in dedicated modules (for example `app/core/<topic>.py` or `app/<domain>/helpers_<topic>.py`), not inline in route/service files.
- Do not add single-use helper abstractions; extract only when reused or when it materially improves readability/testability.
- Python file size policy:
  - target around 300 LOC for new or heavily edited files;
  - do not exceed 500 LOC;
  - when a file grows, split into submodules/directories and treat the directory as the module boundary.
- Existing >500 LOC hotspots to split further when touched:
  - `app/ingestion/pdf_pipeline/artifact_builders.py`
  - `app/retrieval/retriever.py`
  - `app/services/ingest_service.py`
  - `app/ingestion/pdf_pipeline/chunk_builder.py`
  - `app/scripts/run_enterprise_rag_eval.py`
  - `app/ingestion/pdf_pipeline/adapters.py`
  - `app/ingestion/pdf_pipeline/page_structure.py`
- Access model is internal single-tenant; auth/RBAC is intentionally removed.
- Do not assume repo-local `package.json`, `Makefile`, pre-commit config, or CI workflow gates.
