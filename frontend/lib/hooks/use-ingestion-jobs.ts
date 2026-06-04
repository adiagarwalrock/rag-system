"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";

export function useIngestionJobs(clientId: string) {
  return useQuery({
    queryKey: ["ingestion-jobs", clientId],
    queryFn: ({ signal }) => apiClient.listIngestionJobs(clientId, signal),
    enabled: Boolean(clientId),
  });
}

export function useRetryIngestionJob(clientId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, parser }: { jobId: string; parser?: string }) =>
      apiClient.retryIngestionJob(clientId, jobId, parser),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["ingestion-jobs", clientId] }),
  });
}
