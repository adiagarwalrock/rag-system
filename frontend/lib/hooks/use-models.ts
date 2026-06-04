"use client";

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { EmbeddingModelInfo } from "@/lib/api/schemas";

const MODELS_STALE_TIME = 5 * 60 * 1000;

const modelsQuery = {
  queryKey: ["models"] as const,
  queryFn: ({ signal }: { signal: AbortSignal }) => apiClient.listModels(signal),
  staleTime: MODELS_STALE_TIME,
};

export function useModels() {
  return useQuery(modelsQuery);
}

export function useEmbeddingModels() {
  return useQuery({
    ...modelsQuery,
    select: (data) => {
      const providers = data.configured_providers ?? [];
      const all = data.embedding_models ?? [];
      // Only surface models whose provider has an API key configured.
      // If the server returned no configured_providers at all (older backend),
      // fall back to showing everything so the UI is never empty.
      return providers.length > 0
        ? all.filter((m) => providers.includes(m.provider))
        : all;
    },
  });
}

/** Resolve a display label for an embedding model ID from the registry. */
export function resolveEmbedLabel(
  modelId: string | null | undefined,
  embeddingModels: EmbeddingModelInfo[],
): string | null {
  const entry =
    embeddingModels.find((m) => m.id === modelId) ??
    embeddingModels.find((m) => m.default);
  return entry?.display_name || modelId || null;
}
