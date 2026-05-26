# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
cp .env.example .env && ./setup.sh   # requires uv + npm; installs deps + global @llamaindex/liteparse

# Infrastructure
docker-compose up -d qdrant          # required before ingestion/retrieval/chat

# Run the app
uv run streamlit run streamlit_app.py --server.port 8502   # primary UI
uv run uvicorn api:app --reload --port 8000                # optional REST API

# Tests
uv run pytest                                              # full suite (uses in-memory SQLite; no Qdrant/Snowflake needed)
uv run pytest tests/test_retrieval.py                     # single file
uv run pytest tests/test_retrieval.py::test_name          # single test

# Integration readiness (stricter than runtime; requires Snowflake + Qdrant + AI key)
uv run python -m app.scripts.setup_check

# Enterprise RAG evaluation
uv run python -m app.scripts.run_enterprise_rag_eval      # defaults in EVALUATION.md

# Linting / formatting
uv run black .
uv run isort .
uv run ruff check .
```

## Architecture

Single Python monolith with two entry points sharing one service layer:

- **`streamlit_app.py`** — primary UI, loads multipage routes from `ui/pages/*`
- **`api.py`** — optional FastAPI REST surface, routes in `app/api/routes_*.py`

Both call the same orchestration in `app/services/*` in-process. Streamlit uses `ui/lib/api.py` (`VecteraCore`) and does **not** make internal HTTP calls.

### Layers and ownership

| Layer | Location |
|---|---|
| UI pages | `ui/pages/*` |
| UI adapter | `ui/lib/api.py` (`VecteraCore`) |
| REST routes | `app/api/routes_*.py` (thin wrappers only) |
| Orchestration / business logic | `app/services/*` |
| Ingestion & parsing | `app/ingestion/*` |
| Retrieval pipeline | `app/retrieval/*` |
| Vector store | `app/indexing/vector_store.py`, `app/indexing/chat_history_store.py` |
| Relational DB / session factory | `app/db/*` |
| Config / AI init | `app/core/config.py`, `app/core/ai_provider.py` |

### Ingestion flow

1. Upload creates `Document` + `IngestionJob` in `queued` state; raw file saved to `data/raw`.
2. `IngestionQueueManager` workers (`app/services/ingest_queue.py`) process jobs asynchronously.
3. **4-level parser fallback** (each level tried only if its key is set; failure falls through):
   1. **Reducto** (`app/ingestion/parser/external/reducto.py`) — if `ENABLE_EXTERNAL_PARSER=true` and `REDUCTO_API_KEY` set
   2. **LlamaParse** (`app/ingestion/parser/external/llamacloud.py`) — if `ENABLE_EXTERNAL_PARSER=true` and `LLAMA_CLOUD_API_KEY` set
   3. **Layout-aware PDF** (`app/ingestion/parser/custom/pdf_pipeline/`) — PDFs when `ENABLE_LAYOUT_AWARE_PDF=true`
   4. **Legacy** (`app/ingestion/parser/custom/legacy.py`) — always available
4. External parsers (1 & 2) emit `[[START OF PAGE n]]` / `[[END OF PAGE n]]` markers; `to_llama_docs()` splits these into one `LlamaDocument` per page.
5. All paths feed a single `SemanticSplitterNodeParser` + LLM enrichment pass in `app/services/ingest_service.py`.
6. Version metadata resolved into `DocumentVersion`; nodes indexed to Qdrant + SQL mapping in `VectorNodeRegistry`.
7. Status transitions: `queued → processing → indexed / failed`.

### Query / answer flow

1. `ChatConversationService` manages session creation and message persistence.
2. `VecteraRetriever` performs client-scoped hybrid Qdrant search (dense + sparse, with dense fallback).
3. Optional query expansion for comparative/visual/conflict prompts.
4. Cross-encoder reranker (`app/retrieval/cross_encoder_reranker.py`) + semantic/temporal/structural adjustments.
5. `ConflictDetector` flags numeric disagreements across sources.
6. Citations built from selected evidence; LLM synthesis produces grounded answer.
7. Q/A pairs embedded into a dedicated chat-history Qdrant collection for cross-session semantic memory.

## Configuration

Settings are loaded from `.env` via `pydantic-settings` (`app/core/config.py`). Key points:

- `AI_API_KEY` accepts aliases: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`.
- `validate_runtime_settings()` fails on missing/placeholder key — no silent fallback.
- Relational DB: Snowflake when `SNOWFLAKE_ACCOUNT` + `SNOWFLAKE_USER` are set; otherwise SQLite (`rag_local.db`).
- Schema is ensured at runtime via `ensure_runtime_schema()` — keep all schema changes additive.
- Qdrant vector dimensions are enforced; changing `VECTOR_DIMENSIONS` on an existing collection may require recreation.
- Default collection: `rag_collection_oai_parser_extractor`; chat history: `chat_history_v1`.

## Documentation and References

Before implementing or fixing anything that touches a library, framework, or external API used in this repo, look up current documentation — training data may be stale. Use available tools:

- **Context7** (`mcp__claude_ai_Context7__resolve-library-id` + `mcp__claude_ai_Context7__query-docs`) — preferred for library/SDK/framework docs (LlamaIndex, Qdrant client, FastAPI, Pydantic, SQLAlchemy, Streamlit, OpenAI SDK, `sentence-transformers`, `fastembed`, `reductoai`, `llama-cloud`, etc.).
- **Web search** (`mcp__claude_ai_Tavily__tavily_search` / `tavily_research`) — for anything not covered by Context7: changelog entries, GitHub issues, provider-specific API behavior (Reducto, LlamaParse, Qdrant Cloud), model names/pricing, or when Context7 returns insufficient results.

When to use them:
- Calling a library API you haven't verified in this session (e.g. Qdrant client methods, LlamaIndex node parsers, OpenAI responses mode).
- Checking a model identifier or endpoint behavior (default models in `config.py` may drift).
- Debugging an error that looks like a version mismatch or deprecated interface.
- Implementing a new integration against any external service.

## Streamlit ↔ API Parity

Both entry points (`streamlit_app.py` via `VecteraCore` and `api.py` via `app/api/routes_*.py`) must expose the same capabilities. Any feature added or changed on one side **must be reflected on the other**:

- A new service method in `app/services/*` must get both a REST route in `app/api/routes_*.py` **and** a corresponding method in `ui/lib/api.py` (`VecteraCore`).
- A new UI workflow in `ui/pages/*` that calls `VecteraCore` must have an equivalent REST endpoint so external callers can do the same thing.
- Parameter signatures, option flags, and response shapes should match between the two surfaces. If the API adds a query param or request field, the UI adapter should pass it through (even if the UI doesn't yet expose it as a control).
- Removing or renaming a capability on one side requires the same change on the other — do not leave dead routes or orphaned `VecteraCore` methods.

## Engineering Rules

- **Business logic belongs in `app/services/*`** — API routes and UI pages must stay thin.
- **Cache expensive objects** — LLM/embeddings via `app/core/ai_provider.py` (`get_llm`, `get_embeddings`), DB session factory via `app/db/snowflake.py` module globals, Qdrant managers as runtime singletons.
- **File size target ~300 LOC, hard cap 500 LOC.** When a file grows, split into submodules. Files already known to exceed this limit and candidates for splitting when touched: `artifact_builders.py`, `retriever.py`, `ingest_service.py`, `chunk_builder.py`, `run_enterprise_rag_eval.py`, `adapters.py`, `page_structure.py`.
- **When changing a module**, verify dependent contracts/callers across schemas, service interfaces, metadata expectations, and runtime wiring to keep the whole codebase in sync.
- Place utilities in dedicated modules (`app/core/<topic>.py` or `app/<domain>/helpers_<topic>.py`), not inline in route/service files.
- Runtime is internal single-tenant; auth/RBAC is intentionally removed — do not add it back.
- Tests use in-memory SQLite fixtures (`tests/conftest.py`) and do not require Snowflake or Qdrant.
