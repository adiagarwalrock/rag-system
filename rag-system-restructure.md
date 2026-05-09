# Project Restructuring Plan

This plan details how we will map the current application structure to the `production-ai-app` standard requested by the user. The goal is to increase accessibility, separate similar functionality, and make code maintainability easier.

## User Review Required

> [!WARNING]
> This represents a major structural shift. File imports, script paths, and Docker configurations will all need to be updated. Please confirm if you approve of the targeted mapping below before we proceed.

## Proposed Changes

### Root / Application Entrypoints

We intend to keep everything cleanly separated by responsibilities. 

#### [MODIFY] api.py -> app/main.py
#### [NEW] app/models.py (consolidated from `app/schemas/*`)
#### [MODIFY] streamlit_app.py -> frontend/app.py 

---

### Frontend

The UI components currently reside in `ui/`. This maps to the `frontend/` directory in the target structure.

#### [MODIFY] ui/* -> frontend/*

---

### Core Services -> `app/services/`
The core business logic and state management are mapped into the `services/` namespace.

#### [MODIFY] app/services/query_service.py -> app/services/rag_pipeline.py
#### [MODIFY] app/services/chat_conversation_service.py -> app/services/conversation.py
#### [MODIFY] app/services/chat_context_service.py -> app/services/conversation.py
#### [MODIFY] app/retrieval/query_expansion.py -> app/services/query_rewriter.py
*Other existing services like ingest_service will either remain in services or map to specific areas depending on their role.*

---

### Components -> `app/components/`
The custom retrieval layers are isolated here.

#### [MODIFY] app/retrieval/retriever.py -> app/components/hybrid_retriever.py
#### [MODIFY] app/retrieval/reranker.py -> app/components/reranker.py

---

### Database / Ingestion Pipeline -> `data/` or `app/services/`
We will re-evaluate `app/ingestion/` to align with the new model (e.g. `data/raw`, `data/processed`). Scripts to manage migrations go to `scripts/`.

#### [MODIFY] app/scripts/* -> scripts/*
#### [MODIFY] init_db.py -> scripts/seed.py

---

### Evaluation -> `evaluation/`
The current `app/evals` maps cleanly to the top-level `evaluation/` directory.

#### [MODIFY] app/evals/* -> evaluation/*

---

### Prompts -> `app/prompts/`
#### [MODIFY] app/core/prompts.py -> app/prompts/templates.py
#### [MODIFY] app/retrieval/prompt_builder.py -> app/prompts/registry.py

---

### Agents -> `app/agents/`
#### [MODIFY] app/core/ai_provider.py -> app/agents/tools/ 
*(Or a similar restructuring where AI-centric generation tasks become Agents)*

---

### Observability
#### [MODIFY] app/core/token_budget.py -> observability/cost_tracker.py

---

## Verification Plan

Because this is a massive import-breaking change, our primary verification method will be ensuring the code compiles and passes its test suite.

### Automated Tests
1. **Pytest Integration**: Run `pytest tests/` after each major set of file moves and import refactors. 
2. **Type Checking**: Run `mypy` or the existing `type_check_results.txt` scripts to ensure static imports are correct.

### Manual Verification
1. Inspect the new folder structure with `tree` commands to ensure it visually matches the target `production-ai-app` image.
2. Spin up the FastAPI and Streamlit entrypoints (`app/main.py` and `frontend/app.py`) locally to confirm they can still find their relative imports.
