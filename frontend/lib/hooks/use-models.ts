"use client";

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: ({ signal }) => apiClient.listModels(signal),
    staleTime: 5 * 60 * 1000,
  });
}
