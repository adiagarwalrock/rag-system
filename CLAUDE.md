# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
cp .env.example .env && ./setup.sh   # installs Python deps via uv + npm global @llamaindex/liteparse
docker-compose up -d qdrant

# Run
uv run streamlit run streamlit_app.py --server.port 8502
uv run uvicorn api:app --reload --port 8000

# Test
uv run pytest
uv run pytest tests/test_retrieval.py
uv run pytest tests/test_retrieval.py::test_name

# Format
uv run ruff format .
uv run ruff check --fix .

# Type check
uv run ty check

# Integration check (strict: requires Snowflake + Qdrant + valid AI key)
uv run python -m app.scripts.setup_check
```

## Architecture

Two entry points, one service layer:

- `streamlit_app.py` → `ui/pages/*` → `ui/lib/api.py` (`RAGCore`) → `app/services/*` (in-process, no HTTP hop)
- `api.py` → `app/api/routes_*` → `app/services/*` (thin route wrappers)

Business logic lives exclusively in `app/services/`. Routes and UI pages must stay thin.

Layer boundaries:

| Layer | Location |
| --- | --- |
| Orchestration | `app/services/` |
| Ingestion / parsing | `app/ingestion/` |
| Retrieval pipeline | `app/retrieval/` |
| Vector store | `app/indexing/` |
| Relational DB / sessions | `app/db/` |

### Ingestion flow

Upload → `Document` + `IngestionJob` (queued) → raw file saved to `data/raw/` → `IngestionQueueManager` workers → PDF layout pipeline (`app/ingestion/pdf_pipeline/`) or legacy parser → `DocumentVersion` + supersession → Qdrant index + `VectorNodeRegistry` → status transitions (`queued → processing → indexed/failed`).

### Query / chat flow

`ChatConversationService` → `execute_query` → `RAGRetriever` (hybrid dense+sparse Qdrant, client-scoped) → optional query expansion → reranker (semantic + temporal + structural) → conflict detector → citation builder → LLM synthesis → persisted logs.

Cross-session memory: Q/A pairs are embedded into a separate `chat_history` Qdrant collection and injected into future query context.

## Runtime Wiring

- Settings load from `.env` via `pydantic-settings` (`app/core/config.py`). `AI_API_KEY` is the fallback; provider-specific keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`) take precedence.
- `validate_runtime_settings()` fails at startup on a missing or placeholder key — no mock fallback.
- DB: Snowflake when `SNOWFLAKE_ACCOUNT` + `SNOWFLAKE_USER` are set; otherwise SQLite `rag_local.db`.
- Schema is migration-less; `ensure_runtime_schema()` is called at startup — keep schema changes additive-safe.
- Qdrant vector dimensions are enforced at collection creation; changing `VECTOR_DIMENSIONS` or `EMBEDDING_OUTPUT_DIMENSION` on an existing collection requires recreation.
- AI/embedding clients are cached module-globals in `app/core/ai_provider.py` — do not reinstantiate per call.

## Engineering Constraints

- **File size**: target ~300 LOC; hard cap 500 LOC. When a file grows, split into a subpackage directory. Current >500 LOC hotspots to split when touched: `artifact_builders.py`, `retriever.py`, `ingest_service.py`, `chunk_builder.py`, `run_enterprise_rag_eval.py`, `adapters.py`, `page_structure.py`.
- Tests use in-memory SQLite fixtures (`tests/conftest.py`); most tests do not require live Snowflake or Qdrant.
- No `Makefile`, `package.json`, pre-commit config, or CI workflow gates exist in this repo.
- Access model is internal single-tenant; auth/RBAC is intentionally absent.

## Code Style and Design Philosophy

### Object-Oriented Design

Write fully object-oriented Python. Use the complete OOP toolkit deliberately:

- **Dataclasses / Pydantic models** for value objects and DTOs — never plain dicts crossing layer boundaries.
- **Abstract base classes** (`abc.ABC`, `@abstractmethod`) to define contracts for swappable components (parsers, retrievers, synthesizers, store backends).
- **`__slots__`** on hot-path, frequently-instantiated classes to reduce memory overhead.
- **Properties and descriptors** for computed attributes and validated state — not bare attribute access with external mutation.
- **Class methods and static methods** purposefully: `@classmethod` for alternative constructors, `@staticmethod` for pure utilities with no instance/class dependency.
- **Context managers** (`__enter__`/`__exit__` or `contextlib`) for any resource that has acquire/release semantics.
- **`__repr__`, `__eq__`, `__hash__`** on domain objects that appear in logs, sets, or dicts.

### Design Patterns (refactoring.guru)

Apply GoF and enterprise patterns where they remove real duplication or manage real complexity. Preferred patterns in this codebase:

- **Strategy** — swappable algorithms (parser backends, retrieval modes, reranking strategies, LLM providers). Define an ABC, inject via constructor.
- **Factory Method / Abstract Factory** — AI provider construction (`app/core/ai_factory.py`), parser selection. Centralise object creation; callers depend on abstractions.
- **Builder** — multi-step assembly of complex objects (prompt construction, chunk assembly, citation building). Prefer a dedicated `XBuilder` class over long positional argument lists.
- **Observer / Event** — ingestion status transitions, queue notifications. Decouple producers from consumers.
- **Template Method** — shared ingestion/retrieval skeleton with overridable steps. Put the invariant sequence in a base class, let subclasses supply the varying steps.
- **Decorator** — cross-cutting behaviour (logging, timing, retry) added to service methods without modifying them.
- **Repository** — all DB access goes through a repository class; service layer never writes raw SQL or ORM queries inline.
- **Facade** — `RAGCore` (`ui/lib/api.py`) and route handlers are facades; keep them thin and delegate entirely to services.

Avoid pattern overuse: only introduce a pattern when it eliminates a concrete problem (duplication, fragile coupling, untestable logic), not as a structural default.

### Module / File Tree Structure

Prefer **deep, domain-organised hierarchies** over flat directories of files. When adding a new capability:

1. Create a subpackage directory with its own `__init__.py` that exports the public interface.
2. Split internals across focused submodules within that directory (`models.py`, `contracts.py`, `helpers.py`, `pipeline.py`, etc.).
3. Keep the public API surface small — only what external callers need should be re-exported from `__init__.py`.

Example of preferred structure for a new capability:

```text
app/
  ingestion/
    pdf_pipeline/        ← subpackage, not a single pdf_pipeline.py
      __init__.py        ← exports PdfPipeline, PdfPipelineConfig only
      pipeline.py
      chunk_builder.py
      adapters.py
      contracts.py
      models.py
      helpers.py
```

Flat dumps of many peer files in one directory are a refactor signal.
