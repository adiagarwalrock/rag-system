"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";

export function useIngestionJobs(clientId: string) {
  return useQuery({
    queryKey: ["ingestion-jobs", clientId],
    queryFn: ({ signal }) => apiClient.listIngestionJobs(clientId, signal),
    enabled: Boolean(clientId),
    refetchInterval: (query) =>
      query.state.data?.some((job) => ["queued", "processing"].includes(job.status)) ? 4000 : false,
  });
}

export function useRetryIngestionJob(clientId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => apiClient.retryIngestionJob(clientId, jobId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["ingestion-jobs", clientId] }),
  });
}
