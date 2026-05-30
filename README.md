# RAG System

A Python monolith for client-scoped document RAG with two entry points:

- Streamlit app (`streamlit_app.py`) as the primary UI.
- FastAPI app (`api.py`) as an optional REST surface.

Both entry points share the same service layer in `app/services/*`.

## Current Status

- UI and API both use the same in-process business logic (no duplicated workflow code).
- Ingestion runs through a background queue with worker threads.
- Retrieval uses hybrid Qdrant search (dense + sparse) with dense fallback.
- **Agentic RAG** (`ENABLE_AGENTIC_RAG=true`): LangGraph-based multi-pass retrieval loop with LLM evidence evaluation, automatic gap detection, and targeted re-retrieval before synthesis.
- Session-aware chat is enabled, including cross-session semantic memory.
- Access control is removed; runtime is internal single-tenant mode.
- Runtime startup requires a valid OpenAI-compatible API key (`OPENAI_API_KEY` / `AI_API_KEY`).

## Requirements

- Python 3.12+
- `uv`
- Docker + Docker Compose
- `npm` (for global `@llamaindex/liteparse` install during setup)

## Quick Start

Follow these steps from the repository root to get the application running:

1. **Set up environment variables**
   Copy the example environment file and configure the minimum required variables:

   ```bash
   cp .env.example .env
   ```

   *If you are using a minimal configuration, these are the key variables to set:*

   ```bash
   # OpenAI-compatible key (required)
   OPENAI_API_KEY=...
   OPENAI_USE_RESPONSES=true

   # Qdrant connection
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=

   # Optional Snowflake (if omitted, runtime falls back to local SQLite rag_local.db)
   # Note: If using Snowflake, ensure the database and schema are created beforehand.
   SNOWFLAKE_ACCOUNT=...
   SNOWFLAKE_USER=...
   SNOWFLAKE_PASSWORD=...
   SNOWFLAKE_DATABASE=...
   SNOWFLAKE_SCHEMA=PUBLIC
   SNOWFLAKE_WAREHOUSE=...
   SNOWFLAKE_ROLE=...
   ```

2. **Run the setup script**
   This installs Python dependencies via `uv` and necessary npm packages:

   ```bash
   ./setup.sh
   ```

3. **Start Qdrant**
   You can run Qdrant locally via Docker:

   ```bash
   docker-compose up -d qdrant
   ```

   *Alternative:* You can use Qdrant Cloud on their free hosting plan: <https://qdrant.tech/documentation/cloud/>. If using the cloud plan, simply set `QDRANT_URL` and `QDRANT_API_KEY` in your `.env` to match your cluster instead of running the docker command.

4. **Start the Streamlit Application**

   ```bash
   uv run streamlit run streamlit_app.py
   ```

   Streamlit is available at `http://localhost:8501`.

5. **(Optional) Start the API Server**

   ```bash
   uv run uvicorn api:app --reload --port 8000
   ```

   API docs will be available at `http://localhost:8000/docs`.

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
- **Agentic RAG**: `app/agents/*` — LangGraph graph, nodes, state, and adapter
- Vector store integration: `app/indexing/vector_store.py`
- Relational data/session state: `app/db/*`

### Agentic RAG (`app/agents/`)

Enabled via `ENABLE_AGENTIC_RAG=true`. The query path is replaced with a LangGraph state machine:

```
intent_router → vector_retrieval → evidence_evaluator
                      ↑                    │ not sufficient (gap detected)
                      └────────────────────┘
                                           │ sufficient or max iterations
                                           ▼
                           reranker → conflict_detector → citation_builder → synthesizer
```

Key behaviours:
- **Deterministic coverage check**: detects comparison questions and verifies each named entity has ≥ 3 relevant nodes before calling the LLM evaluator.
- **Structured evidence evaluation**: uses OpenAI structured output (`EvidenceEvaluation` Pydantic model) — returns `sufficient`, `gap`, and per-node `node_scores`.
- **Gap-targeted re-retrieval**: on `sufficient=false`, uses the `gap` string as the query for the next Qdrant pass instead of repeating the original question.
- **Node exclusion**: already-seen node IDs are tracked in state; duplicate nodes are filtered in Python after each retrieval pass.
- **Max iterations guard**: `AGENTIC_MAX_ITERATIONS=5` (configurable); forces synthesis after N passes.

Relevant config keys (add to `.env`):
```bash
ENABLE_AGENTIC_RAG=true
AGENTIC_MAX_ITERATIONS=5
AGENTIC_EVIDENCE_EVALUATOR_MODEL=   # defaults to QUERY_EXPANSION_MODEL
```

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
4. Parsing — 4-level fallback chain (each level is skipped if its key is absent or it fails):
   1. **Reducto** — if `ENABLE_EXTERNAL_PARSER=true` and `REDUCTO_API_KEY` is set. VLM-powered agentic table and figure extraction; returns page-delimited markdown.
   2. **LlamaParse** — if `ENABLE_EXTERNAL_PARSER=true` and `LLAMA_CLOUD_API_KEY` / `LLAMAPARSE_API_KEY` is set. Returns page-delimited markdown.
   3. **Layout-aware PDF** — custom pipeline (`app/ingestion/parser/custom/pdf_pipeline/`) for PDFs when `ENABLE_LAYOUT_AWARE_PDF=true`.
   4. **Legacy** — LlamaIndex readers; always available as final fallback.
5. All parsed output goes through a single `SemanticSplitterNodeParser` pass followed by LLM enrichment (title, summary, keywords, questions answered).
6. Version metadata is resolved and persisted (`DocumentVersion` + supersession logic).
7. Nodes are indexed into Qdrant and mapped in `VectorNodeRegistry`.
8. Document/job statuses are updated (`queued` -> `processing` -> `indexed`/`failed`).

### External parser configuration

```bash
# Enable/disable the external parser tier (default: true)
ENABLE_EXTERNAL_PARSER=true

# Reducto (priority 1) — https://reducto.ai
REDUCTO_API_KEY=...

# LlamaParse (priority 2) — https://cloud.llamaindex.ai
LLAMA_CLOUD_API_KEY=...
```

External parsers emit `[[START OF PAGE n]]` / `[[END OF PAGE n]]` markers in their output.
The parser layer splits on these markers so each page becomes one `LlamaDocument` with a
correct `page_num` before the semantic splitter runs.

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

For a dedicated guide on running tests and enterprise evaluation runs, see
[`EVALUATION.md`](./EVALUATION.md).

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


## For serving

Use your static IP by binding both servers to `0.0.0.0` and pointing the frontend API env to the static IP.

Example, replace `YOUR_STATIC_IP`:

```bash
# backend
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

In `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://YOUR_STATIC_IP:8000/api/v1
```

Then run frontend:

```bash
cd frontend
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Open:

```text
http://YOUR_STATIC_IP:3000
```

Also make sure your machine/cloud firewall allows inbound:

```text
TCP 3000  # Next.js UI
TCP 8000  # FastAPI backend
```

Good news: `api.py` already has permissive CORS, so the API should accept requests from `http://YOUR_STATIC_IP:3000`.

For anything beyond local testing, put this behind Nginx/Caddy with HTTPS instead of exposing ports `3000` and `8000` directly.