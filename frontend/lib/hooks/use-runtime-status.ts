"use client";

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";

export function useRuntimeStatus(autoRefresh = false) {
  return useQuery({
    queryKey: ["runtime-status"],
    queryFn: ({ signal }) => apiClient.runtimeStatus(signal),
    refetchInterval: autoRefresh ? 10_000 : false,
  });
}
