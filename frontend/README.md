# RAG Console

Minimal Next.js operations console for the existing RAG backend (`api.py`).

## Prerequisites

- Node.js 18+ and npm
- The FastAPI backend must be running before the console can make API calls:
  ```bash
  # from repo root
  uv run uvicorn api:app --reload --port 8000
  ```

> **Note:** `setup.sh` (run from the repo root) installs a global npm package (`@llamaindex/liteparse`) for the Python pipeline — it does **not** install the frontend dependencies. You must run `npm install` inside `frontend/` separately.

## Install

```bash
cd frontend
cp .env.example .env.local   # sets NEXT_PUBLIC_API_BASE_URL to http://127.0.0.1:8000/api/v1
npm install
```

The default `.env.local` points to the local API server — no changes needed for local development. For a remote or static-IP deployment, update `NEXT_PUBLIC_API_BASE_URL` accordingly.

## Run

```bash
npm run dev
```

Console available at `http://localhost:3000`.

## shadcn/ui assumptions

This app uses the shadcn file layout and design tokens, with local Tailwind components instead of generated backend code. If you want to add generated primitives later:

```bash
npx shadcn@latest add button dialog tabs select switch textarea table
```

## Endpoint notes

The typed API client lives in `lib/api/client.ts`. Endpoint paths are centralized there. It targets the requested nested REST contract and includes compatibility fallbacks for the currently exposed `/api/v1/clients`, `/api/v1/documents`, `/api/v1/query`, and `/api/v1/health` routes.
