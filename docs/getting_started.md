# Getting Started and Project Overview

This document provides a guide to understanding the project's architecture, setting up the development environment, and navigating its core functionalities.

## Project Overview

This project is a Python-based system designed for Retrieval Augmented Generation (RAG). It offers two primary interfaces: a Streamlit UI for interactive use and a FastAPI backend for programmatic access. Both interfaces leverage a shared service layer for core business logic, ensuring consistency across operations.

## Getting Started

Follow these steps to set up the project locally:

1. **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd vectera
    ```

2. **Environment Setup:**
    * **Prerequisites:** Ensure you have `uv` and `npm` installed.
    * **Installation:** Copy the environment example and run the setup script:

        ```bash
        cp .env.example .env
        ./setup.sh
        ```

        This script installs Python dependencies using `uv` and installs global `@llamaindex/liteparse` via `npm`.

3. **Database Setup:**
    * **Qdrant:** For ingestion and retrieval, Qdrant is used as the vector database. Start it in detached mode:

        ```bash
        docker-compose up -d qdrant
        ```

    * **Primary Relational DB:** The application defaults to Snowflake if configured. If Snowflake credentials (`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`) are unset, it falls back to a local SQLite database (`rag_local.db`). The schema is managed at runtime via `ensure_runtime_schema`.

4. **Running the Application:**
    * **Streamlit UI:** This is the primary interface for day-to-day operations.

        ```bash
        uv run streamlit run streamlit_app.py --server.port 8502
        ```

    * **FastAPI (REST API):** For external or programmatic access.

        ```bash
        uv run uvicorn api:app --reload --port 8000
        ```

## Code Structure and Modules

The project follows a structured layout to separate concerns:

* **`app/`**: Contains the core Python logic.
  * **`app/services/`**: Houses the main business logic and orchestration classes.
    * `ingest_service.py`: Orchestrates document ingestion.
    * `query_service.py`: Manages query processing.
    * `chat_conversation_service.py`: Handles session-aware chat interactions.
    * `chat_context_service.py`, `query_history_service.py`: Manage chat history and context.
  * **`app/ingestion/`**: Logic related to data ingestion.
    * `pdf_pipeline/`: Handles advanced PDF parsing.
    * `parser.py`: General document parsing.
    * `metadata_extractor.py`, `validator.py`: Metadata handling.
  * **`app/retrieval/`**: Implements retrieval strategies.
    * `retriever.py`: Orchestrates retrieval.
    * `evidence_selector.py`: Selects relevant evidence.
    * `prompt_builder.py`: Constructs prompts for LLMs.
    * `synthesizer.py`: Generates answers using LLMs.
  * **`app/indexing/`**: Vector store and indexing management.
    * `vector_store.py`: Manages the primary vector store (Qdrant).
    * `chat_history_store.py`: Manages chat memory in Qdrant.
  * **`app/db/`**: Database models, session factory, and persistence logic.
    * `snowflake.py`: Snowflake integration.
    * `schema.py`, `base.py`: Database schema definitions.
  * **`app/api/`**: FastAPI route definitions for the REST API.
    * `routes_clients.py`, `routes_documents.py`, `routes_query.py`, `routes_health.py`
  * **`app/core/`**: Core utilities and configuration.
    * `config.py`: Loads settings from `.env` using Pydantic.
    * `ai_provider.py`: Manages LLM and embedding model initialization.
  * **`app/schemas/`**: Pydantic models for data structures.

* **`ui/`**: Contains the Streamlit application components.
  * **`ui/pages/`**: Individual Streamlit pages (e.g., `1_Chat.py`, `2_Documents.py`).
  * **`ui/lib/api.py`**: The `RAGCore` class acts as an in-process adapter for the Streamlit UI to interact with the service layer, bypassing local HTTP routes.

* **`tests/`**: Unit and integration tests for various components.
  * Includes tests for services, retrieval, ingestion, and database interactions.

* **`docs/`**: Documentation files.
  * `architecture.md`: Current system architecture overview.
  * `architecture.png`, `request_sequence.png`: Architectural diagrams.

## Core Flows

### 1. Document Ingestion

1. **Upload**: A request initiates an `IngestionJob`, storing the raw file in `data/raw`.
2. **Processing**: `IngestionQueueManager` workers process jobs asynchronously.
3. **Parsing**: Documents are parsed using either a layout-aware PDF pipeline or a legacy parser.
4. **Indexing**: Parsed content is converted into nodes, embedded, and indexed into Qdrant. SQL mappings (`VectorNodeRegistry`) are updated.
5. **Completion**: Job status transitions from `queued` to `processing` to `indexed` or `failed`.

### 2. Query and Answer Generation

1. **Query Execution**: The query is processed via `ChatConversationService` or `query_service.execute_query`.
2. **Retrieval**: `RAGRetriever` performs a hybrid search in Qdrant, potentially using query expansion and reranking.
3. **Conflict Detection**: A `ConflictDetector` flags discrepancies in evidence.
4. **Citation**: Citations are constructed from selected evidence.
5. **Synthesis**: An LLM synthesizes a grounded answer.
6. **Logging**: Query, retrieval, and conflict logs are persisted.

### 3. Conversation Memory

1. **Storage**: Chat messages are stored in SQL tables (`chat_sessions`, `chat_messages`).
2. **Summarization**: Session summaries are updated periodically.
3. **Embedding**: Q/A pairs are embedded and stored in a dedicated chat-history Qdrant collection.
4. **Context Injection**: Semantic matches from past conversations can be injected into future query contexts.

### 4. Deletion and Cleanup

* Deleting a document removes associated vectors, SQL records, ingestion jobs, and the raw file.
* Client deletion cascades to clean up related document, query, and chat states, including chat-memory vectors.

## Environment and Integration Notes

* **Configuration**: Settings are loaded from `.env` files via `pydantic-settings` in `app/core/config.py`. API keys (`AI_API_KEY`) accept aliases like `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`.
* **Runtime Validation**: `validate_runtime_settings()` ensures a valid API key is present on startup.
* **Schema Management**: Schema is migration-less and ensured at runtime via `ensure_runtime_schema()` in `api.py` and `ui/lib/api.py`. Additive schema changes are preferred.
* **Vector Dimensions**: Qdrant collection dimensions are strictly enforced. Changing embedding dimensions might require collection recreation.
* **Access Model**: The runtime operates in an internal single-tenant mode; authentication and RBAC are removed.

## Verification

* **Full Test Suite**: Run all tests with `uv run pytest`.
* **Focused Tests**: Execute specific test files or cases, e.g., `uv run pytest tests/test_retrieval.py` or `uv run pytest tests/test_retrieval.py::test_name`.
* **Readiness Check**: Use `uv run python -m app.scripts.setup_check` for a comprehensive check of Snowflake, Qdrant, and AI key configuration. Most tests use in-memory SQLite and do not require Snowflake or Qdrant.

This document aims to provide a solid foundation for understanding and contributing to the project.
