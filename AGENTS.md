# AGENTS.md

## Quick Start (repo root)

- `cp .env.example .env && ./setup.sh` (requires `uv` + `npm`; installs Python deps and global `@llamaindex/liteparse`).
- Start vector DB before ingestion/retrieval: `docker-compose up -d qdrant`.
- Streamlit primary app: `uv run streamlit run streamlit_app.py --server.port 8502`.
- Optional API wrapper: `uv run uvicorn api:app --reload --port 8000`.

## Runtime Wiring

- `streamlit_app.py` is the main UX entrypoint; `ui/pages/*` use `ui/lib/api.py` (`VecteraCore`) to call `app/services/*` directly (no internal HTTP hop).
- `api.py` exposes the same workflows over REST (`/api/v1/*`); keep business logic in `app/services/`, not route handlers/UI pages.
- Boundaries: orchestration `app/services/`, ingestion `app/ingestion/`, retrieval `app/retrieval/`, vector store `app/indexing/vector_store.py`, DB/session `app/db/`.

## Env + Integration Gotchas

- Settings load from `.env` via `pydantic-settings` (`app/core/config.py`); `GOOGLE_API_KEY` and `GEMINI_API_KEY` are aliases.
- Missing/placeholder Google key switches to `MockLLM` + `MockEmbedding`; app still runs but responses are mock-quality.
- If `SNOWFLAKE_ACCOUNT` and `SNOWFLAKE_USER` are unset, runtime falls back to SQLite `rag_local.db` (`app/db/snowflake.py`).
- `api.py` imports models and runs `Base.metadata.create_all(bind=engine)` at startup, so tables auto-create.
- Qdrant vector dimensions are enforced; changing `VECTOR_DIMENSIONS`/`EMBEDDING_OUTPUT_DIMENSION` against an existing collection can require recreating that collection.

## Auth + Seeding

- `AUTH_ENABLED=false` returns stub user `dev-user` from `app/core/dependencies.py`.
- For auth-enabled local testing, seed roles/admin with `uv run python -m app.scripts.seed_admin` (`admin@user.local` / `admin123`).

## Verification

- Full integration check: `uv run python -m app.scripts.setup_check`.
- `setup_check` is strict (Snowflake + Qdrant + valid Google key all required to pass), even though runtime can work with SQLite fallback.
- Test suite: `uv run pytest`.
- Focused tests: `uv run pytest tests/test_retrieval.py` or `uv run pytest tests/test_retrieval.py::test_name`.
- Tests use in-memory SQLite fixtures (`tests/conftest.py`), so they generally do not need Snowflake/Qdrant.

## Engineering Bar For Agents

- Use OOP concepts for new core flows: prefer cohesive classes/objects over procedural sprawl.
- Do not add single-use helper functions/abstractions; extract only when reused or when it materially improves readability/testability.
- Behave like staff-level engineering code is under peer review: clear boundaries, strong naming, typed contracts, and robust error handling.
- Extend the existing architecture patterns (service-layer orchestration, schema-driven API boundaries) instead of bypassing them.
- Do not invent project gates that do not exist here (no repo-local `package.json`, `Makefile`, pre-commit config, or CI workflow).
