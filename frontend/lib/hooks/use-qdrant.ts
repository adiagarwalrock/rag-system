"use client";

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";

export function useQdrantCollections() {
  return useQuery({
    queryKey: ["qdrant-collections"],
    queryFn: ({ signal }) => apiClient.listQdrantCollections(signal),
  });
}

export function useQdrantPoints(collection: string) {
  return useQuery({
    queryKey: ["qdrant-points", collection],
    queryFn: ({ signal }) => apiClient.listQdrantPoints(collection, signal),
    enabled: Boolean(collection),
  });
}
