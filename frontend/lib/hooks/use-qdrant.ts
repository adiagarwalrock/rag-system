"use client";

import { useQuery } from "@tanstack/react-query";
import { apiClient, type QdrantDocumentFilters, type QdrantNodeFilters } from "@/lib/api/client";

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

export function useQdrantNodes(collection: string, filters: QdrantNodeFilters) {
  return useQuery({
    queryKey: ["qdrant-nodes", collection, filters],
    queryFn: ({ signal }) => apiClient.listQdrantNodes(collection, filters, signal),
    enabled: Boolean(collection),
  });
}

export function useQdrantNodeCompare(collection: string, pointIds: string[]) {
  return useQuery({
    queryKey: ["qdrant-node-compare", collection, pointIds],
    queryFn: ({ signal }) => apiClient.compareQdrantNodes(collection, pointIds, signal),
    enabled: Boolean(collection && pointIds.length >= 2 && pointIds.length <= 4),
  });
}

export function useQdrantDocuments(collection: string, filters: QdrantDocumentFilters) {
  return useQuery({
    queryKey: ["qdrant-documents", collection, filters],
    queryFn: ({ signal }) => apiClient.listQdrantDocuments(collection, filters, signal),
    enabled: Boolean(collection),
  });
}

export function useQdrantDocumentCompare(collection: string, documentKeys: string[]) {
  return useQuery({
    queryKey: ["qdrant-document-compare", collection, documentKeys],
    queryFn: ({ signal }) => apiClient.compareQdrantDocuments(collection, documentKeys, signal),
    enabled: Boolean(collection && documentKeys.length >= 2 && documentKeys.length <= 4),
  });
}
