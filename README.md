# RAG System

A Python monolith for client-scoped document RAG with two entry points:

- Streamlit app (`streamlit_app.py`) as the primary UI.
- FastAPI app (`api.py`) as an optional REST surface.

Both entry points share the same service layer in `app/services/*`.

## Current Status

- UI and API both use the same in-process business logic (no duplicated workflow code).
- Ingestion runs through a background queue with worker threads.
- Retrieval uses hybrid Qdrant search (dense + sparse) with dense fallback.
- Session-aware chat is enabled, including cross-session semantic memory.
- Access control is removed; runtime is internal single-tenant mode.
- Runtime startup requires a valid OpenAI-compatible API key (`OPENAI_API_KEY` / `AI_API_KEY`).

## Requirements

- Python 3.12+
- `uv`
- Docker + Docker Compose
- `npm` (for global `@llamaindex/liteparse` install during setup)

## Quick Start

From repo root:

```bash
cp .env.example .env
./setup.sh
docker-compose up -d qdrant
uv run streamlit run streamlit_app.py --server.port 8502
```

Streamlit is available at `http://localhost:8502`.

Optional API server:

```bash
uv run uvicorn api:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

## Runtime Wiring

- `streamlit_app.py` loads multipage UI routes from `ui/pages/*`.
- UI pages call `ui/lib/api.py` (`VecteraCore`), which invokes `app/services/*` directly.
- There is no internal HTTP hop between Streamlit and business services.
- `api.py` exposes the same workflows over REST via `app/api/routes_*`.

## Architecture and Boundaries

See [architecture.md](./architecture.md) for the full system map. Current boundaries are:

- Orchestration: `app/services/*`
- Ingestion/parsing: `app/ingestion/*`
- Retrieval: `app/retrieval/*`
- Vector store integration: `app/indexing/vector_store.py`
- Relational data/session state: `app/db/*`

## Environment and Integrations

Configuration is loaded from `.env` via `pydantic-settings` (`app/core/config.py`).

### API key behavior

- `AI_API_KEY` accepts aliases including `OPENAI_API_KEY`.
- Placeholder or missing keys fail startup validation (`validate_runtime_settings`).
- AI provider initialization is centralized in `app/core/ai_provider.py`.

### Database behavior

- If `SNOWFLAKE_ACCOUNT` and `SNOWFLAKE_USER` are set, SQLAlchemy uses Snowflake.
- Otherwise, runtime falls back to local SQLite (`rag_local.db`).
- `api.py` ensures schema at startup with `ensure_runtime_schema(engine)`.

### Qdrant behavior

- Main collection uses `settings.COLLECTION_NAME` (default `rag_collection_oai`).
- Retrieval prefers hybrid dense+sparse mode; dense fallback is automatic.
- Vector dimension mismatches are enforced and can require collection recreation.

## Ingestion Flow (Current)

1. Upload is validated and persisted (`Document`, `IngestionJob`) in queued state.
2. Raw file is saved to `data/raw`.
3. Background workers process queued ingestion tasks.
4. Parsing:
   - PDF uses layout-aware pipeline by default (with legacy fallback unless strict mode is enabled).
   - DOCX/PPTX and fallback paths use the legacy parser.
5. Version metadata is resolved and persisted (`DocumentVersion` + supersession logic).
6. Nodes are indexed into Qdrant and mapped in `VectorNodeRegistry`.
7. Document/job statuses are updated (`queued` -> `processing` -> `indexed`/`failed`).

## Retrieval and Chat Flow (Current)

1. Queries are scoped by `client_id`.
2. Optional query expansion is applied for comparative/visual/conflict prompts.
3. Retrieval runs in hybrid mode with dense fallback.
4. Deterministic reranking applies semantic + temporal + structural metadata signals.
5. Conflict detection flags numeric disagreements across relevant sources.
6. Citations are generated from selected evidence nodes.
7. Grounded answer synthesis runs via configured OpenAI mode.
8. Query logs, retrieval logs, and conflict logs are persisted.

Chat orchestration (`ChatConversationService`) adds:

- Session creation and message persistence.
- Session summary refresh.
- Cross-session semantic memory using a dedicated Qdrant chat-history collection.

## Document Lifecycle Operations

### Retry ingestion

- Available for `failed`, `indexed`, or `completed` documents.
- Requires original raw file to still exist in `data/raw`.

UI: Document Library -> `Retry`  
API: `POST /api/v1/documents/{document_id}/retry`

### Delete document

- Removes vectors, relational mappings, ingestion jobs, and raw file.
- On failure, status is set to `deleting_failed`.

UI: Document Library -> `Delete`  
API: `DELETE /api/v1/documents/{document_id}?hard=true`

## Verification

Full integration setup check:

```bash
uv run python -m app.scripts.setup_check
```

The setup check is strict and expects all three to pass:

- Snowflake connectivity
- Qdrant connectivity
- OpenAI key validation

Test suite:

```bash
uv run pytest
```

Focused tests:

```bash
uv run pytest tests/test_retrieval.py
uv run pytest tests/test_retrieval.py::test_name
```

## Known Constraints

- Startup validation fails without a real AI API key.
- `setup_check` can fail even when runtime fallback to SQLite is acceptable.
- Changing embedding/vector dimensions against an existing Qdrant collection may require recreating it.
- Chat session management is currently exposed through the internal Streamlit adapter (`ui/lib/api.py`), while REST query endpoints accept `session_id` but do not provide dedicated session CRUD routes.
