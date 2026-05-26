# RAG Console

Minimal Next.js operations console for the existing RAG backend.

## Install

```bash
cd frontend
cp .env.example .env.local
npm install
```

Set `NEXT_PUBLIC_API_BASE_URL` to the mounted API prefix. With the current FastAPI app this is usually:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

## Run

```bash
npm run dev
```

## shadcn/ui assumptions

This app uses the shadcn file layout and design tokens, with local Tailwind components instead of generated backend code. If you want to add generated primitives later:

```bash
npx shadcn@latest add button dialog tabs select switch textarea table
```

## Endpoint notes

The typed API client lives in `lib/api/client.ts`. Endpoint paths are centralized there. It targets the requested nested REST contract and includes compatibility fallbacks for the currently exposed `/api/v1/clients`, `/api/v1/documents`, `/api/v1/query`, and `/api/v1/health` routes.
