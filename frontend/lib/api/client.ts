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
  parserListResponseSchema,
  qdrantCollectionSchema,
  qdrantDocumentCompareResponseSchema,
  qdrantDocumentListResponseSchema,
  qdrantNodeCompareResponseSchema,
  qdrantNodeListResponseSchema,
  qdrantPointSchema,
  qualityRunSchema,
  qualityTestSchema,
  modelListResponseSchema,
  queryHistoryItemSchema,
  queryRequestSchema,
  queryResponseSchema,
  runtimeStatusSchema,
  type ChatSession,
  type ChatMessage,
  type Client,
  type Document,
  type IngestionJob,
  type ParserListResponse,
  type QdrantCollection,
  type QdrantDocumentCompareItem,
  type QdrantDocumentListResponse,
  type QdrantNode,
  type QdrantNodeListResponse,
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

export type QdrantNodeFilters = {
  clientId?: string;
  documentId?: string;
  documentTitle?: string;
  parserName?: string;
  chunkType?: string;
  pageNum?: number;
  search?: string;
  limit?: number;
};

export type QdrantDocumentFilters = {
  clientId?: string;
  documentTitle?: string;
  parserName?: string;
  search?: string;
  limit?: number;
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
  const value = raw as Record<string, unknown>;
  return clientSchema.parse({
    ...value,
    document_count: toFiniteNumber(value.document_count) ?? 0,
    query_count: toFiniteNumber(value.query_count) ?? 0,
    session_count: toFiniteNumber(value.session_count) ?? 0,
    memory_point_count: toFiniteNumber(value.memory_point_count) ?? 0,
  });
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

function toFiniteNumber(value: unknown): number | undefined {
  const numberValue = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

function normalizeImageUrl(url: unknown): string | undefined {
  if (typeof url !== "string" || !url.trim()) return undefined;
  try {
    return new URL(url, apiBaseUrl).toString();
  } catch {
    return url;
  }
}

function normalizeImageAssets(
  rawAssets: unknown,
  citation: Record<string, unknown>,
) {
  if (!Array.isArray(rawAssets)) return [];
  return rawAssets
    .map((raw) => {
      const asset = raw as Record<string, unknown>;
      const url = normalizeImageUrl(asset.url);
      if (!url) return undefined;
      return {
        ...asset,
        url,
        filename: String(asset.filename ?? "image"),
        page_num: toFiniteNumber(asset.page_num ?? citation.page_num ?? citation.page),
        document_id: asset.document_id ?? citation.document_id,
        document_name: asset.document_name ?? citation.document_name ?? citation.filename,
        source_artifact_id: asset.source_artifact_id ?? citation.source_artifact_id,
        source_artifact_type: asset.source_artifact_type ?? citation.source_artifact_type,
      };
    })
    .filter(Boolean);
}

function normalizeCitation(raw: unknown) {
  const value = raw as Record<string, unknown>;
  const score =
    typeof value.score === "number" && Number.isFinite(value.score)
      ? value.score
      : undefined;
  return citationSchema.parse({
    ...value,
    filename:
      value.filename ??
      value.document_name ??
      value.name ??
      value.citation_label ??
      "Unknown source",
    page: value.page ?? value.page_num ?? value.slide_num,
    page_num: value.page_num ?? value.page ?? value.slide_num,
    chunk_id: value.chunk_id ?? value.node_id,
    quote: value.quote ?? value.text,
    score,
    image_assets: normalizeImageAssets(value.image_assets, value),
  });
}

function normalizeConflict(raw: unknown) {
  const item = raw as Record<string, unknown>;
  const rawType = String(item.type ?? item.conflict_type ?? "").toLowerCase();
  const type = rawType.includes("numeric")
    ? "numeric"
    : rawType.includes("version")
      ? "version"
      : rawType.includes("definition")
        ? "definition"
        : "factual";
  return {
    type,
    severity: item.severity ?? "medium",
    explanation: String(item.explanation ?? item.summary ?? "Conflict detected."),
    documents: Array.isArray(item.documents)
      ? item.documents.map(String)
      : undefined,
    values: Array.isArray(item.values) ? item.values.map(String) : undefined,
  };
}

function normalizeStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String).filter(Boolean) : [];
}

function normalizeNumberRecord(value: unknown): Record<string, number> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(
    Object.entries(value)
      .map(([key, raw]) => [key, toFiniteNumber(raw)] as const)
      .filter((entry): entry is readonly [string, number] => entry[1] !== undefined),
  );
}

function normalizeRetrievalTrace(
  value: Record<string, unknown>,
  request: QueryRequest | undefined,
  citations: ReturnType<typeof normalizeCitation>[],
) {
  const rawRetrieval = isRecord(value.retrieval) ? value.retrieval : {};
  const diagnostics = isRecord(value.retrieval_diagnostics)
    ? value.retrieval_diagnostics
    : {};
  const rawImagesUsed = Array.isArray(value.images_used) ? value.images_used : [];
  const imagesUsedCount =
    toFiniteNumber(rawRetrieval.images_used_count) ??
    toFiniteNumber(value.images_used_count) ??
    rawImagesUsed.length;
  const imageEvidenceCount =
    toFiniteNumber(rawRetrieval.image_evidence_count) ??
    toFiniteNumber(value.image_evidence_count) ??
    toFiniteNumber(diagnostics.evidence_image_chunk_count);

  return {
    ...rawRetrieval,
    mode:
      rawRetrieval.mode ??
      value.retrieval_mode ??
      request?.retrieval_mode ??
      "dense_only",
    sparse_available: rawRetrieval.sparse_available ?? value.sparse_available,
    fallback_reason: rawRetrieval.fallback_reason ?? value.fallback_reason,
    selected_chunks: citations,
    query_expanded: Boolean(
      rawRetrieval.query_expanded ?? value.query_expanded ?? false,
    ),
    intent_labels: normalizeStringArray(
      rawRetrieval.intent_labels ?? value.intent_labels ?? diagnostics.intent_labels,
    ),
    companion_queries: normalizeStringArray(
      rawRetrieval.companion_queries ?? value.companion_queries,
    ),
    companion_counts_by_query: normalizeNumberRecord(
      rawRetrieval.companion_counts_by_query ?? value.companion_counts_by_query,
    ),
    image_referenced: Boolean(
      rawRetrieval.image_referenced ?? value.image_referenced ?? imagesUsedCount > 0,
    ),
    image_evidence_count: imageEvidenceCount,
    images_used_count: imagesUsedCount,
    image_assets_used: normalizeImageAssets(
      rawRetrieval.image_assets_used ?? value.image_assets_used,
      {},
    ),
    ranked_image_chunk_count:
      toFiniteNumber(rawRetrieval.ranked_image_chunk_count) ??
      toFiniteNumber(diagnostics.ranked_image_chunk_count),
    evidence_image_chunk_count:
      toFiniteNumber(rawRetrieval.evidence_image_chunk_count) ??
      toFiniteNumber(diagnostics.evidence_image_chunk_count),
    source_count:
      toFiniteNumber(rawRetrieval.source_count) ?? toFiniteNumber(value.source_count),
    evidence_count:
      toFiniteNumber(rawRetrieval.evidence_count) ?? toFiniteNumber(value.evidence_count),
    ranked_document_count:
      toFiniteNumber(rawRetrieval.ranked_document_count) ??
      toFiniteNumber(diagnostics.ranked_document_count),
    evidence_document_count:
      toFiniteNumber(rawRetrieval.evidence_document_count) ??
      toFiniteNumber(diagnostics.evidence_document_count),
    ranked_chunk_types:
      rawRetrieval.ranked_chunk_types ?? diagnostics.ranked_chunk_types,
    evidence_chunk_types:
      rawRetrieval.evidence_chunk_types ?? diagnostics.evidence_chunk_types,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function decodeEscapedAnswer(value: string): string {
  try {
    return JSON.parse(`"${value.replace(/"/g, '\\"')}"`);
  } catch {
    return value
      .replace(/\\n/g, "\n")
      .replace(/\\t/g, "\t")
      .replace(/\\'/g, "'");
  }
}

function extractPythonStyleAnswerEnvelope(
  value: string,
): Record<string, unknown> | undefined {
  const singleQuoted = value.match(/['"]answer['"]\s*:\s*'((?:\\.|[^'\\])*)'/);
  if (singleQuoted?.[1] !== undefined) {
    return { answer: decodeEscapedAnswer(singleQuoted[1]) };
  }

  const doubleQuoted = value.match(/['"]answer['"]\s*:\s*"((?:\\.|[^"\\])*)"/);
  if (doubleQuoted?.[1] !== undefined) {
    return { answer: decodeEscapedAnswer(doubleQuoted[1]) };
  }

  return undefined;
}

function parseAnswerEnvelope(rawAnswer: unknown): {
  answer: string;
  envelope?: Record<string, unknown>;
} {
  if (isRecord(rawAnswer)) {
    return {
      answer: typeof rawAnswer.answer === "string" ? rawAnswer.answer : "",
      envelope: rawAnswer,
    };
  }

  if (typeof rawAnswer !== "string") {
    return {
      answer: rawAnswer === null || rawAnswer === undefined ? "" : String(rawAnswer),
    };
  }

  const trimmed = rawAnswer.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
    return { answer: rawAnswer };
  }

  try {
    const parsed = JSON.parse(trimmed);
    if (isRecord(parsed) && typeof parsed.answer === "string") {
      return { answer: parsed.answer, envelope: parsed };
    }
  } catch {
    const envelope = extractPythonStyleAnswerEnvelope(trimmed);
    if (envelope && typeof envelope.answer === "string") {
      return { answer: envelope.answer, envelope };
    }
  }

  return { answer: rawAnswer };
}

function normalizeQueryResponse(raw: unknown, request?: QueryRequest): QueryResponse {
  const value = raw as Record<string, unknown>;
  const answerEnvelope = parseAnswerEnvelope(value.answer);
  const rawCitations = Array.isArray(value.citations)
    ? value.citations
    : Array.isArray(answerEnvelope.envelope?.citations)
      ? answerEnvelope.envelope.citations
      : [];
  const citations = rawCitations.length
    ? rawCitations.map(normalizeCitation)
    : [];
  const conflicts = Array.isArray(value.conflicts)
    ? value.conflicts.map(normalizeConflict)
    : [];

  return queryResponseSchema.parse({
    ...value,
    id: value.id ?? value.query_id ?? crypto.randomUUID(),
    answer: answerEnvelope.answer,
    citations,
    conflicts,
    retrieval: normalizeRetrievalTrace(value, request, citations),
    memory_hits: value.memory_hits ?? [],
    latency_ms: value.latency_ms ?? 0,
    source_count: toFiniteNumber(value.source_count) ?? 0,
    evidence_count: toFiniteNumber(value.evidence_count) ?? citations.length,
    images_used: Array.isArray(value.images_used) ? value.images_used.map(String) : [],
    image_evidence_count: toFiniteNumber(value.image_evidence_count) ?? 0,
    reasoning_effort: value.reasoning_effort ?? request?.reasoning_effort ?? "medium",
    reasoning: value.reasoning ?? answerEnvelope.envelope?.reasoning,
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
  const citations = Array.isArray(value.citations)
    ? value.citations.map(normalizeCitation)
    : [];
  const conflicts = Array.isArray(value.conflicts)
    ? value.conflicts.map(normalizeConflict)
    : Array.from({ length: Number(value.conflict_count ?? 0) }).map(() => ({
        type: "factual",
        severity: "medium",
        explanation: "Conflict details are available in the query logs.",
      }));
  return queryHistoryItemSchema.parse({
    id: value.id ?? value.query_id,
    question: value.question ?? "",
    answer: value.answer ?? "",
    client_id: value.client_id ?? "",
    session_id: value.session_id,
    citations,
    conflicts,
    retrieval: normalizeRetrievalTrace(
      {
        ...value,
        retrieval: {
          ...(isRecord(value.retrieval) ? value.retrieval : {}),
          top_k: toFiniteNumber(value.retrieval_count),
        },
      },
      undefined,
      citations,
    ),
    latency_ms: Number(value.latency_ms ?? 0),
    source_count: toFiniteNumber(value.source_count) ?? 0,
    evidence_count: toFiniteNumber(value.evidence_count) ?? citations.length,
    images_used: Array.isArray(value.images_used) ? value.images_used.map(String) : [],
    image_evidence_count: toFiniteNumber(value.image_evidence_count) ?? 0,
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

  async listModels(signal?: AbortSignal) {
    return requestJson("/health/models", modelListResponseSchema, {
      signal,
      fallbackPaths: ["/health/models/"],
    });
  },

  async listClients(signal?: AbortSignal) {
    const raw = await requestJson("/clients/", z.array(z.unknown()), {
      signal,
      fallbackPaths: ["/clients"],
    });
    return raw.map(normalizeClient);
  },

  async createClient(input: { name: string; description?: string; embedding_model?: string }, signal?: AbortSignal) {
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

  async uploadDocument(clientId: string, file: File, parserPreference?: string, signal?: AbortSignal) {
    const form = new FormData();
    form.append("file", file);
    form.append("client_id", clientId);
    if (parserPreference && parserPreference !== "auto") {
      form.append("parser", parserPreference);
    }
    const raw = await requestJson("/documents/ingest", z.unknown(), {
      method: "POST",
      body: form,
      signal,
    });
    return normalizeDocument(raw);
  },

  async listParsers(signal?: AbortSignal): Promise<ParserListResponse> {
    return requestJson("/documents/parsers", parserListResponseSchema, { signal });
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

  async retryDocument(clientId: string, documentId: string, parser?: string, signal?: AbortSignal) {
    const url = `/documents/${documentId}/retry${parser ? `?parser=${encodeURIComponent(parser)}` : ""}`;
    const raw = await requestJson(url, z.unknown(), { method: "POST", signal });
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
        current_phase: doc.status,
        version: doc.version,
        vector_points_created: doc.vector_points,
        embedding_model: doc.embedding_model ?? undefined,
        created_at: doc.uploaded_at,
        updated_at: doc.updated_at,
        error: doc.last_error,
      });
    });
  },

  async retryIngestionJob(clientId: string, jobId: string, parser?: string, signal?: AbortSignal) {
    const documentId = jobId.endsWith("-job") ? jobId.slice(0, -4) : jobId;
    return this.retryDocument(clientId, documentId, parser, signal);
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

  async deleteQueryHistoryItem(clientId: string, queryId: string, signal?: AbortSignal) {
    return requestJson(`/clients/${clientId}/history/${queryId}`, z.unknown(), {
      method: "DELETE",
      signal,
      fallbackPaths: [
        `/clients/${clientId}/query-history/${queryId}`,
      ],
    });
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

  async listQdrantNodes(
    collection: string,
    filters: QdrantNodeFilters = {},
    signal?: AbortSignal,
  ): Promise<QdrantNodeListResponse> {
    const params = new URLSearchParams();
    if (filters.clientId) params.set("client_id", filters.clientId);
    if (filters.documentId) params.set("document_id", filters.documentId);
    if (filters.documentTitle) params.set("document_title", filters.documentTitle);
    if (filters.parserName) params.set("parser_name", filters.parserName);
    if (filters.chunkType) params.set("chunk_type", filters.chunkType);
    if (filters.pageNum !== undefined && Number.isFinite(filters.pageNum)) {
      params.set("page_num", String(filters.pageNum));
    }
    if (filters.search) params.set("search", filters.search);
    params.set("limit", String(filters.limit ?? 200));
    const query = params.toString();
    const raw = await requestJson(
      `/qdrant/collections/${encodeURIComponent(collection)}/nodes${query ? `?${query}` : ""}`,
      z.unknown(),
      { signal },
    );
    return qdrantNodeListResponseSchema.parse(raw);
  },

  async compareQdrantNodes(
    collection: string,
    pointIds: string[],
    signal?: AbortSignal,
  ): Promise<QdrantNode[]> {
    const params = new URLSearchParams();
    for (const pointId of pointIds) params.append("point_ids", pointId);
    const raw = await requestJson(
      `/qdrant/collections/${encodeURIComponent(collection)}/nodes/compare?${params.toString()}`,
      z.unknown(),
      { signal },
    );
    const response = qdrantNodeCompareResponseSchema.parse(raw);
    return response.nodes;
  },

  async listQdrantDocuments(
    collection: string,
    filters: QdrantDocumentFilters = {},
    signal?: AbortSignal,
  ): Promise<QdrantDocumentListResponse> {
    const params = new URLSearchParams();
    if (filters.clientId) params.set("client_id", filters.clientId);
    if (filters.documentTitle) params.set("document_title", filters.documentTitle);
    if (filters.parserName) params.set("parser_name", filters.parserName);
    if (filters.search) params.set("search", filters.search);
    params.set("limit", String(filters.limit ?? 200));
    const query = params.toString();
    const raw = await requestJson(
      `/qdrant/collections/${encodeURIComponent(collection)}/documents${query ? `?${query}` : ""}`,
      z.unknown(),
      { signal },
    );
    return qdrantDocumentListResponseSchema.parse(raw);
  },

  async compareQdrantDocuments(
    collection: string,
    documentKeys: string[],
    signal?: AbortSignal,
  ): Promise<QdrantDocumentCompareItem[]> {
    const params = new URLSearchParams();
    for (const documentKey of documentKeys) params.append("document_keys", documentKey);
    const raw = await requestJson(
      `/qdrant/collections/${encodeURIComponent(collection)}/documents/compare?${params.toString()}`,
      z.unknown(),
      { signal },
    );
    const response = qdrantDocumentCompareResponseSchema.parse(raw);
    return response.documents;
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
