import { z } from "zod";
import { ApiError } from "@/lib/api/errors";
import { readStreamingResponse, type StreamHandlers } from "@/lib/api/streaming";
import {
  chatSessionSchema,
  chatMessageSchema,
  citationSchema,
  clientSchema,
  documentSchema,
  ingestionJobSchema,
  qdrantCollectionSchema,
  qdrantPointSchema,
  qualityRunSchema,
  qualityTestSchema,
  queryHistoryItemSchema,
  queryRequestSchema,
  queryResponseSchema,
  runtimeStatusSchema,
  type ChatSession,
  type ChatMessage,
  type Client,
  type Document,
  type IngestionJob,
  type QdrantCollection,
  type QdrantPoint,
  type QualityRun,
  type QualityTest,
  type QueryHistoryItem,
  type QueryRequest,
  type QueryResponse,
  type RuntimeStatus,
} from "@/lib/api/schemas";

const jsonHeaders = { "content-type": "application/json" };
const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000/api/v1";

type RequestOptions = RequestInit & {
  fallbackPaths?: string[];
};

async function requestJson<T>(
  path: string,
  schema: z.ZodType<T>,
  options: RequestOptions = {},
): Promise<T> {
  const paths = [path, ...(options.fallbackPaths ?? [])];
  let lastError: ApiError | undefined;

  for (const candidate of paths) {
    try {
      const response = await fetch(`${apiBaseUrl}${candidate}`, {
        ...options,
        headers: {
          ...(options.body instanceof FormData ? {} : jsonHeaders),
          ...options.headers,
        },
      });
      const body = await parseBody(response);

      if (!response.ok) {
        throw new ApiError({
          message: friendlyHttpMessage(response.status, candidate, body),
          status: response.status,
          path: candidate,
          body,
        });
      }

      const parsed = schema.safeParse(body);
      if (!parsed.success) {
        throw new ApiError({
          message: `API response did not match the expected schema for ${candidate}.`,
          status: response.status,
          path: candidate,
          body,
          validation: parsed.error,
        });
      }

      return parsed.data;
    } catch (error) {
      lastError = toApiError(error, candidate);
      if (!shouldTryFallback(lastError)) break;
    }
  }

  throw lastError ?? new ApiError({ message: "Request failed.", path });
}

async function parseBody(response: Response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function shouldTryFallback(error: ApiError) {
  return error.status === 404 || error.status === 405 || error.status === undefined;
}

function toApiError(error: unknown, path: string) {
  if (error instanceof ApiError) return error;
  if (error instanceof TypeError) {
    return new ApiError({
      message: `Could not connect to API at ${apiBaseUrl}${path}.`,
      path,
      body: String(error),
    });
  }
  if (error instanceof Error) {
    return new ApiError({ message: error.message, path });
  }
  return new ApiError({ message: "Unknown API failure.", path, body: error });
}

function friendlyHttpMessage(status: number, path: string, body: unknown) {
  if (typeof body === "object" && body && "detail" in body) {
    return `Request to ${path} failed (${status}): ${JSON.stringify(body.detail)}`;
  }
  if (typeof body === "string" && body.trim()) {
    return `Request to ${path} failed (${status}): ${body}`;
  }
  return `Request to ${path} failed with status ${status}.`;
}

function normalizeClient(raw: unknown): Client {
  return clientSchema.parse(raw);
}

function normalizeDocument(raw: unknown): Document {
  const value = raw as Record<string, unknown>;
  return documentSchema.parse({
    ...value,
    filename: value.filename ?? value.name ?? "Untitled document",
    uploaded_at: value.uploaded_at ?? value.created_at ?? new Date().toISOString(),
    version: value.version ?? value.version_label ?? 1,
    document_family:
      value.document_family ?? value.version_group ?? value.name ?? "default",
    vector_points: value.vector_points ?? value.vector_point_count,
  });
}

function normalizeCitation(raw: unknown) {
  const value = raw as Record<string, unknown>;
  return citationSchema.parse({
    ...value,
    filename:
      value.filename ??
      value.document_name ??
      value.name ??
      value.citation_label ??
      "Unknown source",
    page: value.page ?? value.page_num ?? value.slide_num,
    chunk_id: value.chunk_id ?? value.node_id,
    quote: value.quote ?? value.text,
  });
}

function normalizeQueryResponse(raw: unknown, request?: QueryRequest): QueryResponse {
  const value = raw as Record<string, unknown>;
  const citations = Array.isArray(value.citations)
    ? value.citations.map(normalizeCitation)
    : [];
  const conflicts = Array.isArray(value.conflicts)
    ? value.conflicts.map((conflict) => {
        const item = conflict as Record<string, unknown>;
        return {
          type: item.type ?? "factual",
          severity: item.severity ?? "medium",
          explanation: String(item.explanation ?? item.summary ?? "Conflict detected."),
          documents: Array.isArray(item.documents)
            ? item.documents.map(String)
            : undefined,
          values: Array.isArray(item.values) ? item.values.map(String) : undefined,
        };
      })
    : [];

  return queryResponseSchema.parse({
    ...value,
    id: value.id ?? value.query_id ?? crypto.randomUUID(),
    answer: value.answer ?? "",
    citations,
    conflicts,
    retrieval: {
      mode: value.retrieval_mode ?? request?.retrieval_mode ?? "dense_only",
      sparse_available: value.sparse_available,
      fallback_reason: value.fallback_reason,
      selected_chunks: citations,
    },
    memory_hits: value.memory_hits ?? [],
    latency_ms: value.latency_ms ?? 0,
    reasoning_effort: value.reasoning_effort ?? request?.reasoning_effort ?? "medium",
    reasoning: value.reasoning,
    created_at: value.created_at ?? new Date().toISOString(),
    session_id: value.session_id,
    user_message_id: value.user_message_id,
    assistant_message_id: value.assistant_message_id,
    raw: value,
  });
}

function normalizeChatMessage(raw: unknown): ChatMessage {
  const value = raw as Record<string, unknown>;
  const result = value.result as Record<string, unknown> | undefined;
  return chatMessageSchema.parse({
    ...value,
    result: result
      ? {
          ...result,
          citations: Array.isArray(result.citations)
            ? result.citations.map(normalizeCitation)
            : [],
        }
      : undefined,
  });
}

function normalizeHistoryItem(raw: unknown): QueryHistoryItem {
  const value = raw as Record<string, unknown>;
  return queryHistoryItemSchema.parse({
    id: value.id ?? value.query_id,
    question: value.question ?? "",
    answer: value.answer ?? "",
    client_id: value.client_id ?? "",
    session_id: value.session_id,
    citations: [],
    conflicts: Array.from({ length: Number(value.conflict_count ?? 0) }).map(() => ({
      type: "factual",
      severity: "medium",
      explanation: "Conflict details are available in the query logs.",
    })),
    retrieval: {
      mode: "dense_only",
      top_k: Number(value.retrieval_count ?? 0) || undefined,
    },
    latency_ms: Number(value.latency_ms ?? 0),
    reasoning_effort: value.reasoning_effort ?? "medium",
    created_at: value.created_at ?? new Date().toISOString(),
    raw: value,
  });
}

function normalizeRuntime(raw: unknown, health?: unknown): RuntimeStatus {
  const value = raw as Record<string, unknown>;
  if ("database" in value && "ai_provider" in value) {
    return runtimeStatusSchema.parse(value);
  }

  const database = (value.database ?? {}) as Record<string, unknown>;
  const qdrant = (value.qdrant ?? {}) as Record<string, unknown>;
  const ai = ((value.ai ?? value.ai_provider) ?? {}) as Record<string, unknown>;
  const parserRoot = (value.parsers ?? {}) as Record<string, unknown>;
  const parserRows = Array.isArray(parserRoot.parsers) ? parserRoot.parsers : [];
  const collections = Array.isArray(qdrant.collections) ? qdrant.collections : [];
  const healthStatus =
    ((health as Record<string, unknown> | null)?.status as string | undefined) ?? "ok";

  return runtimeStatusSchema.parse({
    database: {
      status: normalizeStatus(database.status),
      detail: String(database.message ?? ""),
      metadata: database,
    },
    qdrant: {
      status: normalizeStatus(qdrant.status),
      detail: String(qdrant.message ?? ""),
      metadata: qdrant,
    },
    ai_provider: {
      status: normalizeStatus(ai.status),
      detail: String(ai.message ?? ""),
      metadata: ai,
    },
    parsers: parserRows.map((parser) => {
      const row = parser as Record<string, unknown>;
      return {
        name: parserName(row.name),
        configured: Boolean(row.configured),
        enabled: Boolean(row.enabled),
        priority: Number(row.priority ?? 0),
        latest_status: normalizeParserStatus(row.status ?? row.latest_status),
        latest_error: row.latest_error ? String(row.latest_error) : undefined,
      };
    }),
    collections: collections.map((collection) => {
      const row = collection as Record<string, unknown>;
      return {
        name: String(row.name ?? row.collection_name ?? "unknown"),
        role: row.role ?? "other",
        point_count: Number(row.point_count ?? row.points_count ?? 0),
        vector_type: row.vector_type,
        health: normalizeStatus(row.health ?? row.status),
        last_updated: row.last_updated,
      };
    }),
    api: {
      status: normalizeStatus(healthStatus),
      detail: "REST API connectivity",
      metadata: { base_url: apiBaseUrl, health },
    },
  });
}

function normalizeStatus(value: unknown) {
  const status = String(value ?? "ok").toLowerCase();
  if (status === "ok") return "ok";
  if (status === "error") return "error";
  return "degraded";
}

function normalizeParserStatus(value: unknown) {
  const status = String(value ?? "unused").toLowerCase();
  if (status === "ok" || status === "error" || status === "degraded") return status;
  return "unused";
}

function parserName(value: unknown) {
  const name = String(value ?? "legacy").toLowerCase();
  if (name.includes("reducto")) return "reducto";
  if (name.includes("llama")) return "llamaparse";
  if (name.includes("layout")) return "layout_pdf";
  return "legacy";
}

export const apiClient = {
  baseUrl: apiBaseUrl,

  async health(signal?: AbortSignal) {
    return requestJson("/health/", z.unknown(), {
      signal,
      fallbackPaths: ["/health"],
    });
  },

  async runtimeStatus(signal?: AbortSignal) {
    const [raw, health] = await Promise.allSettled([
      requestJson("/health/status", z.unknown(), { signal, fallbackPaths: ["/health/status/"] }),
      this.health(signal),
    ]);
    const rawValue = raw.status === "fulfilled" ? raw.value : {};
    const healthValue = health.status === "fulfilled" ? health.value : { status: "error" };
    return normalizeRuntime(rawValue, healthValue);
  },

  async listClients(signal?: AbortSignal) {
    const raw = await requestJson("/clients/", z.array(z.unknown()), {
      signal,
      fallbackPaths: ["/clients"],
    });
    return raw.map(normalizeClient);
  },

  async createClient(input: { name: string; description?: string }, signal?: AbortSignal) {
    const raw = await requestJson("/clients/", z.unknown(), {
      method: "POST",
      body: JSON.stringify(input),
      signal,
      fallbackPaths: ["/clients"],
    });
    return normalizeClient(raw);
  },

  async deleteClient(clientId: string, signal?: AbortSignal) {
    return requestJson(`/clients/${clientId}`, z.unknown(), {
      method: "DELETE",
      signal,
    });
  },

  async listDocuments(clientId: string, signal?: AbortSignal) {
    const raw = await requestJson(`/documents/?client_id=${encodeURIComponent(clientId)}`, z.array(z.unknown()), {
      signal,
      fallbackPaths: [`/documents?client_id=${encodeURIComponent(clientId)}`],
    });
    return raw.map(normalizeDocument);
  },

  async uploadDocument(clientId: string, file: File, signal?: AbortSignal) {
    const form = new FormData();
    form.append("file", file);
    form.append("client_id", clientId);
    const raw = await requestJson("/documents/ingest", z.unknown(), {
      method: "POST",
      body: form,
      signal,
    });
    return normalizeDocument(raw);
  },

  async getDocument(clientId: string, documentId: string, signal?: AbortSignal) {
    const raw = await requestJson(`/documents/${documentId}`, z.unknown(), {
      signal,
    });
    return normalizeDocument(raw);
  },

  async deleteDocument(clientId: string, documentId: string, signal?: AbortSignal) {
    return requestJson(`/documents/${documentId}?hard=true`, z.unknown(), {
      method: "DELETE",
      signal,
    });
  },

  async retryDocument(clientId: string, documentId: string, signal?: AbortSignal) {
    const raw = await requestJson(`/documents/${documentId}/retry`, z.unknown(), {
      method: "POST",
      signal,
    });
    return normalizeDocument(raw);
  },

  async listIngestionJobs(clientId: string, signal?: AbortSignal): Promise<IngestionJob[]> {
    const raw = await requestJson(`/documents/?client_id=${encodeURIComponent(clientId)}`, z.array(z.unknown()), {
      signal,
      fallbackPaths: [`/documents?client_id=${encodeURIComponent(clientId)}`],
    });
    return raw.map((item) => {
      const doc = normalizeDocument(item);
      return ingestionJobSchema.parse({
        id: `${doc.id}-job`,
        document_id: doc.id,
        filename: doc.filename,
        client_id: doc.client_id,
        status: doc.status === "deleted" ? "failed" : doc.status,
        parser_attempted: doc.parser_used,
        fallback_stage: doc.status === "failed" ? "fallback exhausted" : "auto fallback",
        current_phase: doc.status,
        version: doc.version,
        vector_points_created: doc.vector_points,
        created_at: doc.uploaded_at,
        updated_at: doc.updated_at,
        error: doc.last_error,
      });
    });
  },

  async retryIngestionJob(clientId: string, jobId: string, signal?: AbortSignal) {
    const documentId = jobId.endsWith("-job") ? jobId.slice(0, -4) : jobId;
    return this.retryDocument(clientId, documentId, signal);
  },

  async listSessions(clientId: string, signal?: AbortSignal): Promise<ChatSession[]> {
    return requestJson(`/clients/${clientId}/sessions`, z.array(chatSessionSchema), {
      signal,
      fallbackPaths: [`/clients/${clientId}/sessions/`],
    });
  },

  async createSession(clientId: string, signal?: AbortSignal) {
    return requestJson(`/clients/${clientId}/sessions`, chatSessionSchema, {
      method: "POST",
      body: JSON.stringify({}),
      signal,
      fallbackPaths: [`/clients/${clientId}/sessions/`],
    });
  },

  async deleteSession(clientId: string, sessionId: string, signal?: AbortSignal) {
    return requestJson(`/clients/${clientId}/sessions/${sessionId}`, z.unknown(), {
      method: "DELETE",
      signal,
    });
  },

  async listSessionMessages(clientId: string, sessionId: string, signal?: AbortSignal): Promise<ChatMessage[]> {
    const raw = await requestJson(
      `/clients/${clientId}/sessions/${sessionId}/messages`,
      z.array(z.unknown()),
      { signal },
    );
    return raw.map(normalizeChatMessage);
  },

  async query(input: QueryRequest, signal?: AbortSignal) {
    const payload = queryRequestSchema.parse(input);
    const raw = await requestJson("/query/", z.unknown(), {
      method: "POST",
      body: JSON.stringify({
        client_id: payload.client_id,
        question: payload.question,
        session_id: payload.session_id,
        reasoning_effort: payload.reasoning_effort,
        reasoning_summary: payload.reasoning_summary,
      }),
      signal,
      fallbackPaths: ["/query"],
    });
    return normalizeQueryResponse(raw, payload);
  },

  async streamQuery(input: QueryRequest, handlers: StreamHandlers<QueryResponse>, signal?: AbortSignal) {
    const payload = queryRequestSchema.parse({ ...input, stream: true });
    const response = await fetch(`${apiBaseUrl}/query/`, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify(payload),
      signal,
    });
    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new ApiError({
        message: `Query stream failed (${response.status})`,
        status: response.status,
        path: "/query/",
        body,
      });
    }
    return readStreamingResponse(response, {
      ...handlers,
      onFinal: (raw) => handlers.onFinal?.(normalizeQueryResponse(raw, payload)),
    });
  },

  async listQueryHistory(clientId: string, signal?: AbortSignal): Promise<QueryHistoryItem[]> {
    const raw = await requestJson(`/clients/${clientId}/history`, z.unknown(), {
      signal,
      fallbackPaths: [
        `/clients/${clientId}/history/`,
        `/clients/${clientId}/query-history`,
        `/clients/${clientId}/query-history/`,
      ],
    });
    const rows = Array.isArray(raw) ? raw : ((raw as { rows?: unknown[] }).rows ?? []);
    return rows.map(normalizeHistoryItem);
  },

  async listQdrantCollections(signal?: AbortSignal): Promise<QdrantCollection[]> {
    const raw = await requestJson("/qdrant/collections", z.array(z.unknown()), { signal });
    return raw.map((item) => qdrantCollectionSchema.parse(item));
  },

  async listQdrantPoints(collection: string, signal?: AbortSignal): Promise<QdrantPoint[]> {
    const raw = await requestJson(
      `/qdrant/collections/${encodeURIComponent(collection)}/points`,
      z.array(z.unknown()),
      { signal },
    );
    return raw.map((item) => qdrantPointSchema.parse(item));
  },

  async getQdrantPoint(collection: string, pointId: string, signal?: AbortSignal) {
    return requestJson(
      `/qdrant/collections/${encodeURIComponent(collection)}/points/${encodeURIComponent(pointId)}`,
      qdrantPointSchema,
      { signal },
    );
  },

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async listQualityTests(_clientId: string, _signal?: AbortSignal): Promise<QualityTest[]> {
    return [];
  },

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async createQualityTest(_clientId: string, _input: Omit<QualityTest, "id" | "client_id" | "created_at">, _signal?: AbortSignal): Promise<never> {
    throw new ApiError({ message: "Quality evaluation REST endpoint is not yet implemented.", path: "/quality/tests", status: 404 });
  },

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async runQualityTest(_clientId: string, _testId: string, _signal?: AbortSignal): Promise<never> {
    throw new ApiError({ message: "Quality evaluation REST endpoint is not yet implemented.", path: "/quality/tests/{test_id}/run", status: 404 });
  },

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async listQualityRuns(_clientId: string, _signal?: AbortSignal): Promise<QualityRun[]> {
    return [];
  },
};
