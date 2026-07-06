"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import { ApiError } from "@/lib/api/errors";
import type { QueryRequest } from "@/lib/api/schemas";

export function useSessions(clientId: string) {
  return useQuery({
    queryKey: ["sessions", clientId],
    queryFn: ({ signal }) => apiClient.listSessions(clientId, signal),
    enabled: Boolean(clientId),
  });
}

export function useCreateSession(clientId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => apiClient.createSession(clientId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["sessions", clientId] }),
  });
}

export function useDeleteSession(clientId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => apiClient.deleteSession(clientId, sessionId),
    onSuccess: (_, sessionId) => {
      queryClient.invalidateQueries({ queryKey: ["sessions", clientId] });
      queryClient.removeQueries({ queryKey: ["session-messages", clientId, sessionId] });
    },
  });
}

export function useDeleteSessions(clientId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (sessionIds: string[]) => {
      const results = await Promise.allSettled(
        sessionIds.map((sessionId) => apiClient.deleteSession(clientId, sessionId)),
      );
      const failure = results.find(
        (result) =>
          result.status === "rejected" &&
          !(result.reason instanceof ApiError && result.reason.status === 404),
      );
      if (failure?.status === "rejected") throw failure.reason;
      return results;
    },
    onSuccess: (_, sessionIds) => {
      queryClient.invalidateQueries({ queryKey: ["sessions", clientId] });
      sessionIds.forEach((sessionId) => {
        queryClient.removeQueries({ queryKey: ["session-messages", clientId, sessionId] });
      });
    },
  });
}

export function useSessionMessages(clientId: string, sessionId: string) {
  return useQuery({
    queryKey: ["session-messages", clientId, sessionId],
    queryFn: ({ signal }) => apiClient.listSessionMessages(clientId, sessionId, signal),
    enabled: Boolean(clientId && sessionId),
  });
}

export function useAskQuery() {
  return useMutation({
    mutationFn: ({ input, signal }: { input: QueryRequest; signal?: AbortSignal }) =>
      apiClient.query(input, signal),
  });
}
