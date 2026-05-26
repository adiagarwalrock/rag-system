"use client";

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";

export function useQueryHistory(clientId: string) {
  return useQuery({
    queryKey: ["history", clientId],
    queryFn: ({ signal }) => apiClient.listQueryHistory(clientId, signal),
    enabled: Boolean(clientId),
  });
}
