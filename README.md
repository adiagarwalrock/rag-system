# RAG System

A comprehensive RAG system featuring an independent, loosely-coupled presentation layer (Streamlit) alongside a discrete REST API (FastAPI) that share a common internal Python services backbone. This design enables the UI to remain incredibly snappy by directly hooking into databases and ingestion logic, while simultaneously allowing for robust REST communication for potential downstream systems.

## Requirements

- Python 3.12+ (Using `uv`)
- Docker & Docker Compose

### Install uv

If `uv` is not installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:

```bash
uv --version
```

## Global Setup

Run all commands from the repository root.

### Quick Start

After cloning, run the setup script to install all dependencies (requires `uv` and `npm`):

```bash
./setup.sh
```

### Manual Setup

1. Create your environment file:

   ```bash
   cp .env.example .env
   ```

2. Update `.env` values:
   - Set all `SNOWFLAKE_*` fields for Snowflake.
   - Set `OPENAI_API_KEY` for real LLM and embedding responses.
   - Keep `QDRANT_URL` as `http://localhost:6333` for local Docker.
   - If `SNOWFLAKE_*` values are omitted, the app uses local SQLite (`rag_local.db`) as a fallback.

3. Install dependencies:

   ```bash
   uv sync
   ```

4. Start local vector database (Qdrant):

   ```bash
   docker-compose up -d qdrant
   ```

5. Run setup checks for Snowflake, Qdrant, and LLM config:

   ```bash
   uv run python -m app.scripts.setup_check
   ```

   This validates:
   - Snowflake connectivity with `SELECT 1`
   - Qdrant connectivity and collection readiness
   - OpenAI API key validity against the OpenAI API

## Running the UI (Streamlit)

The primary UI entry point leverages Streamlit. It directly utilizes internal models and services (`app.services.*`) without executing HTTP requests internally.

```bash
uv run streamlit run streamlit_app.py --server.port 8502
```

*UI will run on <http://localhost:8502>*

## Running the API (FastAPI)

If you wish to interact with Vectera programmatically or build an alternate frontend down the line, the FastAPI interface is separately available.

```bash
uv run uvicorn api:app --reload --port 8000
```

*API will run on <http://localhost:8000>*
*Swagger docs at <http://localhost:8000/docs>*

## Architecture

Please see [architecture.md](./architecture.md) for a comprehensive diagrammatic and structural breakdown of the application architecture.

## Implementation Details

### Database (Snowflake / SQLite)

- The system heavily relies on SQLAlchemy ORM using Snowflake for production, with a seamless fallback to a local SQLite database (`rag_local.db`).
- It tracks relational metadata such as Client workspaces, Document families/versions, Vector registry metadata mappings (mapping Qdrant node IDs to physical documents), and Query Logs to enable robust document management without overloading the vector database with broad document relationship logic.

### Chunking Strategy

- Input documents (PDF, DOCX, PPTX) are processed natively via `LlamaIndex`'s specific file readers in `app/ingestion/parser.py`.
- During parsing, structural data (like page numbers and slide numbers) are extracted and attached to chunks.
- For non-layout-aware documents, the pipeline uses `SemanticSplitterNodeParser` with configurable `SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE` and `SEMANTIC_SPLITTER_BUFFER_SIZE`.
- Layout-aware PDF ingestion keeps its specialized artifact-aware chunking path (tables/charts/figures) and skips this generic splitter stage.
- Optional pipeline enhancements include `TitleExtractor` when API keys are available to fortify LLM metadata contexts.

### Retrieval Approach

Retrieval relies on a specialized, multi-stage retrieval pipeline (`app/retrieval/retriever.py`):

1. **Vector Retrieval**: Starts with client-isolated `ExactMatchFilter` retrieval executing against the Qdrant backend, pulling the top `K+5` nearest neighbors using `text-embedding-3-large`.
2. **Authority & Recency Reranking**: Re-scores outputs pushing documents with higher authority indicators or newer internal version rankings to the top (`reranker.py`).
3. **Temporal Ranking**: Adjusts scores logically based on explicitly resolved `effective_from`/`effective_to` dates relative to the current UTC timestamp, penalizing expired sources (`temporal_ranker.py`).
4. **Citation Building**: Translates standard chunk nodes into explicitly labeled citation structures that are fed directly inside the synthesized generation prompt, referencing source text precisely via `page_num` and `version_label`.

### Handling Document Versioning

- Versioning is deeply ingrained. `version_resolver.py` scans filename and snippet inputs during ingestion to identify patterns corresponding to quarters (e.g., Q1_2024), years, explicit version numbers (v2), and status cues (draft/final).
- Parsed documents are mapped into a unified `document_family`.
- When new, current variants of the same document family are uploaded, older versions are dynamically identified and marked `is_current = False`.
- Non-current versions are penalized during the retrieval phase but kept in the index in case historical contexts are necessary.

### Handling Conflicting Information

- `conflict_detector.py` operates automatically at the end of the retrieval pipeline to flag contradictions spanning retrieved context chunks.
- It clusters retrieved vectors based on the document or version groups and uses regex-based heuristic extractions across source strings to find "numeric disagreements" mapping to identical topics.
- If conflicts are identified (for instance, versions containing different numeric facts referencing similar contexts like quarterly revenue), it builds warning alerts pushed directly into the Streamlit UI, helping operators cross-check the authoritative source rather than receiving obfuscated hallucinations.

### Handling Charts/Tables

- Tables and visual chart data representation are partially handled during parsing via textual heuristic signals.
- `parser.py` flags specific pages/chunks with indicators (`table_detected = True`, `chart_detected = True`) by counting structured text artifacts like tabs, pipemarks (`|`), or references to standard diagram nomenclature ("Figure A", "graph").
- These indicators are embedded as metadata for LlamaIndex pipeline filtering and context awareness but do not currently rebuild tabular markdown extraction or utilize multimodal Computer Vision reading.

### Known Limitations

- **Ingestion Blocking**: Document processing and parsing routines operate synchronously. Uploading large multi-hundred-page files blocks the Streamlit frontend.
- **Tabular Insight Limitations**: Because the system does not utilize Vision-Language Models (VLMs) or advanced OCR, complex nested tables or image-based visual charts cannot be queried effectively. It relies strictly on textual scrape artifacts.
- **Conflict Regex Brittleness**: The numeric extraction algorithm in `conflict_detector.py` handles standard financial phraseology ("X grew by Y%") but misses abstract prose contradictions effectively due to the absence of dedicated LLM contradiction verification.

### What I would improve with more time

- **Asynchronous Task Processing**: Shift the ingestion pipeline (embedding, semantic extractions, vector insertions) into a Celery/Redis queue or background FastAPI task pattern to untether the UI thread.
- **Multimodal Visual Embeddings**: Implement LlamaIndex's vision pipelines using a multimodal OpenAI model to correctly ingest visual graphs and convert bounded tables into raw markdown formats during the ingestion phase for precise layout querying.
- **Advanced Cross-Encoder Reranking**: Swap the basic additive metadata-heuristics reranker for a Neural Cross-Encoder logic model (like Cohere Rerank) that drastically improves top-k contextual sorting over plain embeddings without degrading performance.

## Operations & Document Lifecycle

Documents uploaded to Vectera move through several ingestion states: `processing` -> `indexed` (or `failed`).

### Retrying Failed Documents

If a document upload fails (e.g., due to an API timeout or malformed parser data), the status will be marked as `failed` and an `error_message` will be preserved in the latest `IngestionJob`.

- **Method**: Navigate to the Streamlit UI, open the Document Details panel, and click **Retry Ingestion**. Alternatively, trigger a POST call to `/api/v1/documents/{document_id}/retry`.
- **Note**: The original uploaded raw file must still exist in `data/raw/` in order to retry successfully.

### Deleting Documents

Documents can be permanently retired from the vector search space using the delete feature.

- **Method**: From the Document Details panel, click **Delete Document** and confirm. Alternatively, issue a `DELETE /api/v1/documents/{document_id}?hard=true`.
- **Behavior**: This attempts a coordinated wipe. It removes vector node points from Qdrant, drops associated metadata records and chunks from the database, removes the physical file from `data/raw/`, and deletes the database record entirely. If vector deletion fails partially, the status will downgrade to `deleting_failed` to signal an operator.

**Database Schemas and Status Tracking**: We utilize an implicit string-based enum for statuses which allows for in-place workflow expansion without destructive SQL schema migrations. No manual `CREATE TABLE` alterations are required unless you add entirely new column properties.
