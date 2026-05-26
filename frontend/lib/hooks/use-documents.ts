"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";

export const documentKeys = {
  all: (clientId: string) => ["documents", clientId] as const,
  detail: (clientId: string, documentId: string) => ["documents", clientId, documentId] as const,
};

export function useDocuments(clientId: string) {
  return useQuery({
    queryKey: documentKeys.all(clientId),
    queryFn: ({ signal }) => apiClient.listDocuments(clientId, signal),
    enabled: Boolean(clientId),
  });
}

export function useDocument(clientId: string, documentId: string) {
  return useQuery({
    queryKey: documentKeys.detail(clientId, documentId),
    queryFn: ({ signal }) => apiClient.getDocument(clientId, documentId, signal),
    enabled: Boolean(clientId && documentId),
  });
}

export function useUploadDocument(clientId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ file }: { file: File }) =>
      apiClient.uploadDocument(clientId, file),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: documentKeys.all(clientId) }),
  });
}

export function useDeleteDocument(clientId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (documentId: string) => apiClient.deleteDocument(clientId, documentId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: documentKeys.all(clientId) }),
  });
}

export function useRetryDocument(clientId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (documentId: string) => apiClient.retryDocument(clientId, documentId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: documentKeys.all(clientId) }),
  });
}
