"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

const workspaceKey = "rag-console.workspace";
const effortKey = "rag-console.reasoning-effort";
const llmModelKey = "rag-console.llm-model";
const sessionKey = (workspaceId: string) => `rag-console.session.${workspaceId}`;

type WorkspaceState = {
  workspaceId: string;
  setWorkspaceId: (id: string) => void;
  sessionId: string;
  setSessionId: (id: string) => void;
  reasoningEffort: "low" | "medium" | "high";
  setReasoningEffort: (effort: "low" | "medium" | "high") => void;
  llmModel: string | undefined;
  setLlmModel: (model: string) => void;
};

const WorkspaceContext = createContext<WorkspaceState | null>(null);

export function WorkspaceProvider({ children }: { children: React.ReactNode }) {
  const [workspaceId, setWorkspaceIdState] = useState("");
  const [sessionId, setSessionIdState] = useState("");
  const [reasoningEffort, setReasoningEffortState] = useState<"low" | "medium" | "high">("medium");
  const [llmModel, setLlmModelState] = useState<string | undefined>(undefined);

  useEffect(() => {
    const storedWorkspace = localStorage.getItem(workspaceKey) ?? "";
    setWorkspaceIdState(storedWorkspace);
    setSessionIdState(getStoredSession(storedWorkspace));
    const storedEffort = localStorage.getItem(effortKey);
    if (storedEffort === "low" || storedEffort === "medium" || storedEffort === "high") {
      setReasoningEffortState(storedEffort);
    }
    const storedModel = localStorage.getItem(llmModelKey) ?? undefined;
    setLlmModelState(storedModel);
  }, []);

  const setWorkspaceId = useCallback((id: string) => {
    setWorkspaceIdState(id);
    if (id) localStorage.setItem(workspaceKey, id);
    setSessionIdState(getStoredSession(id));
  }, []);

  const setSessionId = useCallback(
    (id: string) => {
      setSessionIdState(id);
      if (workspaceId) setStoredSession(workspaceId, id);
    },
    [workspaceId],
  );

  const setReasoningEffort = useCallback((effort: "low" | "medium" | "high") => {
    setReasoningEffortState(effort);
    localStorage.setItem(effortKey, effort);
  }, []);

  const setLlmModel = useCallback((model: string) => {
    setLlmModelState(model);
    localStorage.setItem(llmModelKey, model);
  }, []);

  const value = useMemo(
    () => ({
      workspaceId,
      setWorkspaceId,
      sessionId,
      setSessionId,
      reasoningEffort,
      setReasoningEffort,
      llmModel,
      setLlmModel,
    }),
    [workspaceId, setWorkspaceId, sessionId, setSessionId, reasoningEffort, setReasoningEffort, llmModel, setLlmModel],
  );

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}

export function useWorkspaceStore() {
  const context = useContext(WorkspaceContext);
  if (!context) throw new Error("useWorkspaceStore must be used within WorkspaceProvider.");
  return context;
}

export function getStoredSession(workspaceId: string) {
  if (typeof window === "undefined" || !workspaceId) return "";
  return localStorage.getItem(sessionKey(workspaceId)) ?? "";
}

export function setStoredSession(workspaceId: string, sessionId: string) {
  if (!workspaceId) return;
  if (sessionId) {
    localStorage.setItem(sessionKey(workspaceId), sessionId);
  } else {
    localStorage.removeItem(sessionKey(workspaceId));
  }
}
