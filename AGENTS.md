# AGENTS.md

## Quick Start (repo root)

- `cp .env.example .env && ./setup.sh` (`setup.sh` requires `uv` + `npm`, installs Python deps, and installs global `@llamaindex/liteparse`).
- Start Qdrant before ingestion/retrieval/chat: `docker-compose up -d qdrant`.
- Streamlit primary app: `uv run streamlit run streamlit_app.py --server.port 8501` (use `8502` if avoiding a local port collision).
- Optional REST app: `uv run uvicorn api:app --reload --port 8000`.
- Optional Next.js console: `cd frontend && cp .env.example .env.local && npm install && npm run dev`; set `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api/v1`.

## Runtime Wiring

- `streamlit_app.py` and `api.py` share the same service layer; Streamlit uses `ui/lib/api.py` (`VecteraCore`) and does not call local HTTP routes.
- The dedicated Next.js frontend in `frontend/` calls the FastAPI REST surface; keep REST compatibility fallbacks in `frontend/lib/api/client.ts` aligned with `app/api/*`.
- Treat `frontend/` as a first-class app surface, not an afterthought: preserve typed API boundaries, React Query data flow, shadcn/Tailwind conventions, and UX parity with backend capabilities.
- Keep business logic in `app/services/*`; keep API routes (`app/api/*`) and UI pages (`ui/pages/*`) thin.
- Boundaries: orchestration `app/services/`, ingestion/parsing `app/ingestion/`, retrieval `app/retrieval/`, agentic query flow `app/agents/`, vector infra `app/indexing/`, DB/session state `app/db/`.

## Env + Integration Gotchas

- Settings load from `.env` via `pydantic-settings` (`app/core/config.py`); use `OPENAI_API_KEY` for OpenAI-compatible calls. `GEMINI_API_KEY` also accepts `GOOGLE_API_KEY`.
- Startup is strict: `validate_runtime_settings()` fails on missing/placeholder `OPENAI_API_KEY` (no mock fallback).
- If `SNOWFLAKE_ACCOUNT` and `SNOWFLAKE_USER` are unset, runtime falls back to SQLite `rag_local.db` (`app/db/snowflake.py`).
- Schema is migration-less and ensured at runtime via `ensure_runtime_schema(...)` (`api.py`, `ui/lib/api.py`); keep schema changes additive-safe.
- Qdrant vector dimensions are enforced; changing `VECTOR_DIMENSIONS`/`EMBEDDING_OUTPUT_DIMENSION` on an existing collection may require collection recreation.
- Agentic RAG is query-path only and gated by `ENABLE_AGENTIC_RAG`; related knobs include `AGENTIC_MAX_ITERATIONS`, `AGENTIC_EVIDENCE_EVALUATOR_MODEL`, and `AGENTIC_PLANNER_MODEL`.
- Multiple parser backends are intentional. Parser routing is centralized in `app/ingestion/parser/registry.py` plus `app/ingestion/parser/__init__.py`; auto mode currently falls through Reducto -> LlamaParse -> layout-aware PDF -> Docling -> legacy, depending on config and file type.
- External parser keys: `REDUCTO_API_KEY` and `LLAMA_CLOUD_API_KEY`/`LLAMAPARSE_API_KEY`; local parser toggles include `ENABLE_LAYOUT_AWARE_PDF` and `ENABLE_DOCLING_PARSER`.
- OpenAI Responses/reasoning settings (`OPENAI_USE_RESPONSES`, `REASONING_EFFORT`, `REASONING_SUMMARY`, response token/timeouts) live in `app/core/config.py`; preserve streaming callbacks for reasoning and answer deltas.

## Verification

- Full test suite: `uv run pytest`.
- Focused tests: `uv run pytest tests/test_retrieval.py` or `uv run pytest tests/test_retrieval.py::test_name`.
- Agentic graph tests: `uv run pytest tests/agent/test_agent_graph.py`.
- Integration readiness check: `uv run python -m app.scripts.setup_check`.
- `setup_check` is stricter than runtime fallback: Snowflake + Qdrant + valid AI key must all pass.
- Tests use in-memory SQLite fixtures (`tests/conftest.py`), so most tests do not require Snowflake/Qdrant.
- Frontend checks, when touching `frontend/`: `cd frontend && npm run lint && npm run typecheck` after dependencies are installed.

## Agent Persona

- Act as a senior staff FAANG engineer for this repo: design for end-to-end system coherence, prefer sleek low-overhead performance, use abstractions only when they materially help, and keep solutions portable across local, Docker, and CI.

## Engineering Rules For Agents

- Treat the repo as one integrated system, not isolated files: optimize for end-to-end behavior and consistency across `app/`, `ui/`, `frontend/`, API, retrieval, ingestion, and DB layers.
- When changing one module, verify dependent contracts/callers (schemas, service interfaces, metadata expectations, runtime wiring) so the whole codebase stays in sync.
- Engineer changes at a senior staff bar: production-grade design, clear ownership boundaries, measurable performance impact, and maintainability under long-term iteration.
- Prefer sleek, low-overhead code paths: avoid unnecessary allocations/indirection, keep hot paths simple, and choose the most efficient approach that preserves readability.
- Use higher-order/composable functions deliberately where they improve reuse and clarity, but avoid functional abstraction layers that add runtime cost without clear value.
- Keep implementations portable across local dev, Docker, and CI: avoid platform-specific assumptions and favor standard-library/Python-native patterns unless a dependency is justified.
- Prefer clean OOP for new core flows: add cohesive classes/managers/services with explicit ownership, narrow public methods, dependency injection where useful, and no procedural sprawl.
- Design for extensibility and reuse: define typed contracts for pluggable components, keep implementations swappable, avoid copy-paste variants, and extract shared behavior only when it has clear reuse across parsers/services/UI flows.
- Keep Python type annotations complete on new or changed public functions, dataclasses, service methods, parser contracts, and callback signatures; avoid `Any` unless the boundary is genuinely dynamic and document the shape nearby.
- Keep TypeScript strict and explicit in `frontend/`: type API DTOs, component props, hooks, mutation/query results, and form schemas; avoid untyped `any` and duplicated backend response shapes.
- When changing parser selection, update both the registry metadata/availability and the `parse_document(...)` routing branch; keep parser contracts returning `(list[LlamaDocument], list[dict])`.
- For new parsers, implement a small typed adapter around the parser SDK/runtime, register availability/metadata in the registry, preserve normalized retrieval metadata, add focused tests, and avoid embedding parser-specific logic in ingest services.
- When changing query execution, preserve both regular retrieval and agentic retrieval contracts, including citations, conflicts, persisted logs, streaming callbacks, and session-aware chat context.
- Cache expensive/shared objects instead of recreating per call:
  - LLM/embeddings: `app/core/ai_provider.py` (`initialize_ai_provider`, `get_llm`, `get_embeddings`).
  - DB connection/session factory: `app/db/snowflake.py` module globals (`engine`, `SessionLocal`).
  - Qdrant/vector and ingestion runtime objects: `vector_store_manager`, `chat_history_store`, ingestion queue manager.
  - Parser/model-heavy runtime objects such as Docling converters and cross-encoder rerankers.
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
- Do not assume a root-level `package.json`, `Makefile`, pre-commit config, or CI workflow gates; the Next.js package lives under `frontend/`.
