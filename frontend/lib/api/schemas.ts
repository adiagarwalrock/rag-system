import { z } from "zod";

const stringish = z.union([z.string(), z.number()]).transform(String);
const optionalStringish = z
  .union([z.string(), z.number(), z.null(), z.undefined()])
  .transform((value) => (value === null || value === undefined ? undefined : String(value)));
const record = z.record(z.unknown());

export const statusSchema = z.enum(["ok", "degraded", "error"]);
export const documentStatusSchema = z.enum([
  "queued",
  "processing",
  "indexed",
  "failed",
  "deleted",
]);
export const reasoningEffortSchema = z.enum(["low", "medium", "high"]);
export const retrievalModeSchema = z.enum(["hybrid", "dense_only", "auto"]);
export const retrievalTraceModeSchema = z.enum([
  "hybrid",
  "dense_only",
  "dense_fallback",
  "sparse_only",
]);

export const clientSchema = z
  .object({
    id: z.string(),
    name: z.string(),
    description: z.string().nullish(),
    created_at: z.string(),
    document_count: z.number().optional(),
    query_count: z.number().optional(),
    session_count: z.number().optional(),
    memory_point_count: z.number().optional(),
  })
  .passthrough();

export const documentSchema = z
  .object({
    id: z.string(),
    client_id: z.string(),
    filename: z.string(),
    status: documentStatusSchema.catch("queued"),
    document_family: z.string().default("default"),
    version: stringish.default("1"),
    parser_used: z
      .string()
      .nullish()
      .transform((value) => value ?? undefined),
    uploaded_at: z.string(),
    updated_at: z.string().optional(),
    vector_points: z.number().optional(),
    chunk_count: z.number().optional(),
    active_collection: z.string().optional(),
    last_error: z.string().optional(),
    metadata: record.optional(),
  })
  .passthrough();

export const ingestionJobSchema = z
  .object({
    id: z.string(),
    document_id: z.string(),
    filename: z.string(),
    client_id: z.string(),
    status: z.enum(["queued", "processing", "indexed", "failed"]).catch("queued"),
    parser_attempted: z.string().optional(),
    fallback_stage: z.string().optional(),
    current_phase: z.string().optional(),
    version: optionalStringish.optional(),
    vector_points_created: z.number().optional(),
    created_at: z.string(),
    updated_at: z.string().optional(),
    error: z.string().optional(),
  })
  .passthrough();

export const chatSessionSchema = z
  .object({
    id: z.string(),
    client_id: z.string(),
    title: z.string().nullish(),
    created_at: z.string(),
    updated_at: z.string().optional(),
  })
  .passthrough();

export const citationImageAssetSchema = z
  .object({
    url: z.string(),
    filename: z.string(),
    page_num: z.number().nullish(),
    document_id: z.string().nullish(),
    document_name: z.string().nullish(),
    source_artifact_id: z.string().nullish(),
    source_artifact_type: z.string().nullish(),
  })
  .passthrough();

export const citationSchema = z
  .object({
    document_id: z.string().optional(),
    document_name: z.string().optional(),
    filename: z.string().default("Unknown source"),
    page: z.number().optional(),
    page_num: z.number().optional(),
    chunk_id: z.string().optional(),
    quote: z.string().optional(),
    score: z.number().optional(),
    source_artifact_id: z.string().nullish(),
    source_artifact_type: z.string().nullish(),
    image_assets: z.array(citationImageAssetSchema).default([]),
  })
  .passthrough();

export const conflictSchema = z
  .object({
    type: z.enum(["numeric", "factual", "version", "definition"]).catch("factual"),
    severity: z.enum(["low", "medium", "high"]).catch("medium"),
    explanation: z.string(),
    values: z.array(z.string()).optional(),
    documents: z.array(z.string()).optional(),
  })
  .passthrough();

export const retrievalTraceSchema = z
  .object({
    mode: retrievalTraceModeSchema.catch("dense_only"),
    dense_score: z.number().optional(),
    sparse_score: z.number().optional(),
    rerank_score: z.number().optional(),
    top_k: z.number().optional(),
    sparse_available: z.boolean().optional(),
    fallback_reason: z.string().optional(),
    selected_chunks: z.array(citationSchema).optional(),
    query_expanded: z.boolean().optional(),
    intent_labels: z.array(z.string()).default([]),
    companion_queries: z.array(z.string()).default([]),
    companion_counts_by_query: z.record(z.number()).default({}),
    image_referenced: z.boolean().optional(),
    image_evidence_count: z.number().optional(),
    images_used_count: z.number().optional(),
    image_assets_used: z.array(citationImageAssetSchema).default([]),
    ranked_image_chunk_count: z.number().optional(),
    evidence_image_chunk_count: z.number().optional(),
    source_count: z.number().optional(),
    evidence_count: z.number().optional(),
    ranked_document_count: z.number().optional(),
    evidence_document_count: z.number().optional(),
    ranked_chunk_types: record.optional(),
    evidence_chunk_types: record.optional(),
  })
  .passthrough();

export const memoryHitSchema = z
  .object({
    id: z.string(),
    session_id: z.string().nullish(),
    question: z.string(),
    answer_preview: z.string(),
    score: z.number(),
    created_at: z.string().optional(),
  })
  .passthrough();

export const queryRequestSchema = z.object({
  question: z.string().min(1),
  client_id: z.string().min(1),
  session_id: z.string().optional(),
  reasoning_effort: reasoningEffortSchema.optional(),
  reasoning_summary: z.enum(["auto", "concise", "detailed"]).optional(),
  retrieval_mode: retrievalModeSchema.optional(),
  include_memory: z.boolean().optional(),
  include_conflicts: z.boolean().optional(),
  stream: z.boolean().optional(),
});

export const queryResponseSchema = z
  .object({
    id: z.string(),
    answer: z.string(),
    reasoning: z.string().nullish(),
    citations: z.array(citationSchema).default([]),
    conflicts: z.array(conflictSchema).default([]),
    retrieval: retrievalTraceSchema,
    memory_hits: z.array(memoryHitSchema).default([]),
    latency_ms: z.number(),
    source_count: z.number().default(0),
    evidence_count: z.number().default(0),
    images_used: z.array(z.string()).default([]),
    image_evidence_count: z.number().default(0),
    reasoning_effort: reasoningEffortSchema,
    reasoning_summary: z.enum(["auto", "concise", "detailed"]).optional(),
    created_at: z.string(),
    session_id: z.string().nullish(),
    user_message_id: z.string().optional(),
    assistant_message_id: z.string().optional(),
    raw: record.optional(),
  })
  .passthrough();

export const chatMessageSchema = z
  .object({
    id: z.string(),
    client_id: z.string(),
    session_id: z.string(),
    role: z.enum(["user", "assistant"]),
    content: z.string(),
    turn_index: z.number().optional(),
    query_log_id: z.string().nullish(),
    created_at: z.string().optional(),
    result: z
      .object({
        reasoning: z.string().nullish(),
        citations: z.array(citationSchema).default([]),
        query_id: z.string().nullish(),
      })
      .optional(),
  })
  .passthrough();

export const queryHistoryItemSchema = queryResponseSchema
  .omit({ memory_hits: true })
  .extend({
    question: z.string(),
    client_id: z.string(),
    session_id: z.string().nullish(),
    answer: z.string(),
  })
  .passthrough();

export const runtimeComponentStatusSchema = z
  .object({
    status: statusSchema,
    detail: z.string().optional(),
    metadata: record.optional(),
  })
  .passthrough();

export const parserStatusSchema = z
  .object({
    name: z.enum(["reducto", "llamaparse", "layout_pdf", "legacy"]),
    configured: z.boolean(),
    enabled: z.boolean(),
    priority: z.number(),
    latest_status: z.enum(["ok", "degraded", "error", "unused"]),
    latest_error: z.string().optional(),
    average_parse_time_ms: z.number().optional(),
  })
  .passthrough();

export const qdrantCollectionSchema = z
  .object({
    name: z.string(),
    role: z.enum(["documents", "memory", "other"]).catch("other"),
    point_count: z.number(),
    vector_type: z.enum(["dense", "sparse", "hybrid"]).optional(),
    health: statusSchema,
    last_updated: optionalStringish.optional(),
  })
  .passthrough();

export const runtimeStatusSchema = z
  .object({
    database: runtimeComponentStatusSchema,
    qdrant: runtimeComponentStatusSchema,
    ai_provider: runtimeComponentStatusSchema,
    parsers: z.array(parserStatusSchema),
    collections: z.array(qdrantCollectionSchema),
    api: runtimeComponentStatusSchema.optional(),
  })
  .passthrough();

export const qdrantPointSchema = z
  .object({
    id: z.string(),
    collection: z.string(),
    document_id: z.string().optional(),
    filename: z.string().optional(),
    page: z.number().optional(),
    chunk_preview: z.string().optional(),
    dense_vector_size: z.number().optional(),
    sparse_vector_available: z.boolean().optional(),
    document_family: z.string().optional(),
    version: optionalStringish.optional(),
    payload: record.optional(),
    score_breakdown: z.record(z.number()).optional(),
  })
  .passthrough();

export const qdrantNodeSchema = z
  .object({
    id: z.string(),
    collection: z.string(),
    document_id: z.string().nullish(),
    document_name: z.string().nullish(),
    client_id: z.string().nullish(),
    parser_name: z.string().nullish(),
    parser_version: z.string().nullish(),
    chunk_id: z.string().nullish(),
    chunk_type: z.string().nullish(),
    page_num: z.number().nullish(),
    page_nums: z.array(z.number()).default([]),
    section_path: z.string().nullish(),
    citation_label: z.string().nullish(),
    text: z.string().nullish(),
    text_preview: z.string().nullish(),
    text_length: z.number().default(0),
    node_metadata: record.default({}),
    top_level_metadata: record.default({}),
    raw_payload: record.optional(),
  })
  .passthrough();

export const qdrantNodeFacetsSchema = z
  .object({
    documents: z
      .array(
        z
          .object({
            id: z.string().nullish(),
            title: z.string(),
            count: z.number(),
          })
          .passthrough(),
      )
      .default([]),
    parsers: z
      .array(
        z
          .object({
            name: z.string(),
            count: z.number(),
          })
          .passthrough(),
      )
      .default([]),
    chunk_types: z
      .array(
        z
          .object({
            name: z.string(),
            count: z.number(),
          })
          .passthrough(),
      )
      .default([]),
  })
  .passthrough();

export const qdrantNodeListResponseSchema = z
  .object({
    nodes: z.array(qdrantNodeSchema).default([]),
    facets: qdrantNodeFacetsSchema.default({
      documents: [],
      parsers: [],
      chunk_types: [],
    }),
    total: z.number().default(0),
  })
  .passthrough();

export const qdrantNodeCompareResponseSchema = z
  .object({
    nodes: z.array(qdrantNodeSchema).default([]),
  })
  .passthrough();

export const qdrantDocumentCandidateSchema = z
  .object({
    document_key: z.string(),
    collection: z.string(),
    client_id: z.string().nullish(),
    document_id: z.string().nullish(),
    document_name: z.string().nullish(),
    parser_name: z.string().nullish(),
    parser_version: z.string().nullish(),
    node_count: z.number().default(0),
    page_count: z.number().default(0),
    text_length: z.number().default(0),
    preview: z.string().nullish(),
  })
  .passthrough();

export const qdrantDocumentFacetsSchema = z
  .object({
    documents: z
      .array(
        z
          .object({
            id: z.string().nullish(),
            title: z.string(),
            count: z.number(),
          })
          .passthrough(),
      )
      .default([]),
    clients: z
      .array(
        z
          .object({
            id: z.string(),
            count: z.number(),
          })
          .passthrough(),
      )
      .default([]),
    parsers: z
      .array(
        z
          .object({
            name: z.string(),
            count: z.number(),
          })
          .passthrough(),
      )
      .default([]),
  })
  .passthrough();

export const qdrantDocumentListResponseSchema = z
  .object({
    documents: z.array(qdrantDocumentCandidateSchema).default([]),
    facets: qdrantDocumentFacetsSchema.default({
      documents: [],
      clients: [],
      parsers: [],
    }),
    total: z.number().default(0),
  })
  .passthrough();

export const qdrantDocumentSourceNodeSchema = z
  .object({
    id: z.string().nullish(),
    chunk_id: z.string().nullish(),
    chunk_type: z.string().nullish(),
    page_num: z.number().nullish(),
    page_nums: z.array(z.number()).default([]),
    section_path: z.string().nullish(),
    citation_label: z.string().nullish(),
    text_length: z.number().default(0),
    text_preview: z.string().nullish(),
  })
  .passthrough();

export const qdrantDocumentCompareItemSchema = qdrantDocumentCandidateSchema
  .extend({
    markdown: z.string().default(""),
    source_nodes: z.array(qdrantDocumentSourceNodeSchema).default([]),
    metadata_summary: record.default({}),
  })
  .passthrough();

export const qdrantDocumentCompareResponseSchema = z
  .object({
    documents: z.array(qdrantDocumentCompareItemSchema).default([]),
  })
  .passthrough();

export const qualityTestSchema = z
  .object({
    id: z.string(),
    client_id: z.string(),
    name: z.string(),
    question: z.string(),
    expected_answer: z.string().optional(),
    required_citations: z.array(z.string()).optional(),
    expected_conflict: z.boolean().optional(),
    created_at: z.string(),
  })
  .passthrough();

export const qualityRunSchema = z
  .object({
    id: z.string(),
    test_id: z.string(),
    status: z.enum(["pending", "running", "passed", "failed"]),
    retrieval_recall: z.number().optional(),
    citation_match: z.number().optional(),
    answer_faithfulness: z.number().optional(),
    conflict_detection: z.boolean().optional(),
    latency_ms: z.number().optional(),
    notes: z.string().optional(),
    raw: record.optional(),
    created_at: z.string(),
  })
  .passthrough();

export const parserInfoSchema = z.object({
  id: z.string(),
  label: z.string(),
  available: z.boolean(),
  description: z.string(),
});
export const parserListResponseSchema = z.object({
  parsers: z.array(parserInfoSchema),
});

export type ParserInfo = z.infer<typeof parserInfoSchema>;
export type ParserListResponse = z.infer<typeof parserListResponseSchema>;

export type Client = z.infer<typeof clientSchema>;
export type Document = z.infer<typeof documentSchema>;
export type IngestionJob = z.infer<typeof ingestionJobSchema>;
export type ChatSession = z.infer<typeof chatSessionSchema>;
export type ChatMessage = z.infer<typeof chatMessageSchema>;
export type CitationImageAsset = z.infer<typeof citationImageAssetSchema>;
export type Citation = z.infer<typeof citationSchema>;
export type Conflict = z.infer<typeof conflictSchema>;
export type RetrievalTrace = z.infer<typeof retrievalTraceSchema>;
export type MemoryHit = z.infer<typeof memoryHitSchema>;
export type QueryRequest = z.infer<typeof queryRequestSchema>;
export type QueryResponse = z.infer<typeof queryResponseSchema>;
export type QueryHistoryItem = z.infer<typeof queryHistoryItemSchema>;
export type RuntimeStatus = z.infer<typeof runtimeStatusSchema>;
export type QdrantCollection = z.infer<typeof qdrantCollectionSchema>;
export type QdrantPoint = z.infer<typeof qdrantPointSchema>;
export type QdrantNode = z.infer<typeof qdrantNodeSchema>;
export type QdrantNodeListResponse = z.infer<typeof qdrantNodeListResponseSchema>;
export type QdrantNodeCompareResponse = z.infer<typeof qdrantNodeCompareResponseSchema>;
export type QdrantDocumentCandidate = z.infer<typeof qdrantDocumentCandidateSchema>;
export type QdrantDocumentListResponse = z.infer<typeof qdrantDocumentListResponseSchema>;
export type QdrantDocumentCompareItem = z.infer<typeof qdrantDocumentCompareItemSchema>;
export type QdrantDocumentCompareResponse = z.infer<typeof qdrantDocumentCompareResponseSchema>;
export type QualityTest = z.infer<typeof qualityTestSchema>;
export type QualityRun = z.infer<typeof qualityRunSchema>;
