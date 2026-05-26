"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { QualityTest } from "@/lib/api/schemas";

export function useQualityTests(clientId: string) {
  return useQuery({
    queryKey: ["quality-tests", clientId],
    queryFn: ({ signal }) => apiClient.listQualityTests(clientId, signal),
    enabled: Boolean(clientId),
  });
}

export function useQualityRuns(clientId: string) {
  return useQuery({
    queryKey: ["quality-runs", clientId],
    queryFn: ({ signal }) => apiClient.listQualityRuns(clientId, signal),
    enabled: Boolean(clientId),
  });
}

export function useCreateQualityTest(clientId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: Omit<QualityTest, "id" | "client_id" | "created_at">) =>
      apiClient.createQualityTest(clientId, input),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["quality-tests", clientId] }),
  });
}

export function useRunQualityTest(clientId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (testId: string) => apiClient.runQualityTest(clientId, testId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["quality-runs", clientId] });
      queryClient.invalidateQueries({ queryKey: ["quality-tests", clientId] });
    },
  });
}
