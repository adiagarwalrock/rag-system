"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";

export function useQueryHistory(clientId: string) {
  return useQuery({
    queryKey: ["history", clientId],
    queryFn: ({ signal }) => apiClient.listQueryHistory(clientId, signal),
    enabled: Boolean(clientId),
    retry: false,
  });
}

export function useDeleteQueryHistoryItem(clientId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (queryId: string) => apiClient.deleteQueryHistoryItem(clientId, queryId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["history", clientId] }),
  });
}
