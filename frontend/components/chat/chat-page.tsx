"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { ChatMessage as ApiChatMessage, QueryRequest, QueryResponse } from "@/lib/api/schemas";
import { useSessionMessages } from "@/lib/hooks/use-chat";
import { useClients } from "@/lib/hooks/use-clients";
import { useWorkspaceStore } from "@/lib/state/workspace-store";
import { ChatComposer } from "@/components/chat/chat-composer";
import { Check, Copy } from "lucide-react";
import { CitationChip } from "@/components/chat/citation-chip";
import { ErrorState } from "@/components/common/error-state";
import { MarkdownContent } from "@/components/common/markdown-content";
import { StatusBadge } from "@/components/common/status-badge";

type ThreadMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: QueryResponse;
  streaming?: boolean;
};

export default function ChatPage() {
  const queryClient = useQueryClient();
  const clients = useClients();
  const {
    workspaceId,
    setWorkspaceId,
    sessionId,
    setSessionId,
    reasoningEffort,
    setReasoningEffort,
  } = useWorkspaceStore();
  const sessionMessages = useSessionMessages(workspaceId, sessionId);
  const [retrievalMode, setRetrievalMode] = useState<"auto" | "hybrid" | "dense_only">("auto");
  const [includeMemory, setIncludeMemory] = useState(true);
  const [includeConflicts, setIncludeConflicts] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [messages, setMessages] = useState<ThreadMessage[]>([]);
  const [error, setError] = useState<unknown>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!workspaceId && clients.data?.[0]) setWorkspaceId(clients.data[0].id);
  }, [clients.data, setWorkspaceId, workspaceId]);

  useEffect(() => {
    if (!sessionId) {
      setMessages([]);
      return;
    }
    if (sessionMessages.data) {
      setMessages(sessionMessages.data.map(toThreadMessage));
    }
  }, [sessionId, sessionMessages.data]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages, isStreaming]);

  async function applyFallback(
    input: QueryRequest,
    assistantId: string,
    signal: AbortSignal,
  ) {
    const fallback = await apiClient.query(input, signal);
    if (fallback.session_id) {
      setSessionId(fallback.session_id);
      queryClient.invalidateQueries({ queryKey: ["sessions", workspaceId] });
      queryClient.invalidateQueries({
        queryKey: ["session-messages", workspaceId, fallback.session_id],
      });
    }
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? { ...message, content: fallback.answer, response: fallback, streaming: false }
          : message,
      ),
    );
  }

  async function submit(question: string) {
    if (!workspaceId) return;
    const controller = new AbortController();
    const assistantId = crypto.randomUUID();
    abortRef.current = controller;
    setError(null);
    setIsStreaming(true);
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", content: question },
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);

    const input: QueryRequest = {
      question,
      client_id: workspaceId,
      session_id: sessionId || undefined,
      reasoning_effort: reasoningEffort,
      retrieval_mode: retrievalMode,
      include_memory: includeMemory,
      include_conflicts: includeConflicts,
    };

    let finalFromStream: QueryResponse | undefined;
    try {
      await apiClient.streamQuery(
        input,
        {
          onText: (chunk) =>
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + chunk, streaming: true }
                  : message,
              ),
            ),
          onFinal: (final) => {
            finalFromStream = final;
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: final.answer, response: final, streaming: false }
                  : message,
              ),
            );
          },
        },
        controller.signal,
      );
      if (!finalFromStream) {
        await applyFallback(input, assistantId, controller.signal);
      }
    } catch {
      try {
        await applyFallback(input, assistantId, controller.signal);
      } catch (queryError) {
        setError(queryError);
        setMessages((current) => current.filter((message) => message.id !== assistantId));
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="relative flex h-[calc(100vh-3rem)] min-h-[680px] flex-col overflow-hidden">
      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-44 pt-16">
        <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col">
          {messages.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center pb-32 text-center">
              <h1 className="text-2xl font-medium text-foreground">Good to see you.</h1>
              <p className="mt-3 max-w-xl text-sm leading-6 text-muted-foreground">
                Ask across documents in the selected client workspace. Use the plus button in the composer to switch clients.
              </p>
              <button
                className="mt-6 rounded-full border border-border bg-card px-4 py-2 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
                onClick={() => submit("How many customers does Digital Realty have?")}
                disabled={!workspaceId || isStreaming}
              >
                How many customers does Digital Realty have?
              </button>
            </div>
          ) : (
            <div className="flex flex-col gap-7">
              {messages.map((message) => (
                <ChatThreadMessage key={message.id} message={message} />
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </div>

      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-background via-background to-transparent px-3 pb-6 pt-12">
        <div className="mx-auto max-w-3xl">
          {error ? <div className="mb-3"><ErrorState error={error} /></div> : null}
          <ChatComposer
            clients={clients.data ?? []}
            activeClientId={workspaceId}
            onClientChange={setWorkspaceId}
            disabled={!workspaceId || isStreaming}
            streaming={isStreaming}
            onSubmit={submit}
            onCancel={() => abortRef.current?.abort()}
            reasoningEffort={reasoningEffort}
            onReasoningEffortChange={setReasoningEffort}
            retrievalMode={retrievalMode}
            onRetrievalModeChange={setRetrievalMode}
            includeMemory={includeMemory}
            onIncludeMemoryChange={setIncludeMemory}
            includeConflicts={includeConflicts}
            onIncludeConflictsChange={setIncludeConflicts}
          />
          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            RAG Console can make mistakes. Verify answers against cited source material.
          </p>
        </div>
      </div>
    </div>
  );
}

function ChatThreadMessage({
  message,
}: {
  message: ThreadMessage;
}) {
  const [copied, setCopied] = useState(false);

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[82%] rounded-2xl rounded-br-md bg-blue-600 px-4 py-3 text-sm leading-6 text-white">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-border bg-card font-mono text-[11px] text-blue-300">
        AI
      </div>
      <div className="min-w-0 flex-1">
        {message.response ? (
          <div className="flex flex-wrap gap-2">
            <StatusBadge status="ok" label="answered" />
            <StatusBadge status="neutral" label={message.response.retrieval.mode} />
            {message.response.retrieval.sparse_available === false && <StatusBadge status="warning" label="dense-only fallback" />}
            {message.response.conflicts.length ? <StatusBadge status="warning" label={`${message.response.conflicts.length} conflicts`} /> : null}
            <StatusBadge status="neutral" label={`${message.response.latency_ms} ms`} />
            <button
              className="inline-flex h-6 items-center gap-1 rounded-md border border-border bg-muted px-2 text-xs text-muted-foreground hover:text-foreground"
              onClick={async () => {
                await navigator.clipboard.writeText(message.content);
                setCopied(true);
                window.setTimeout(() => setCopied(false), 1200);
              }}
              disabled={!message.content}
            >
              {copied ? <Check className="h-3.5 w-3.5 text-green-300" /> : <Copy className="h-3.5 w-3.5" />}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
        ) : null}
        <MarkdownContent className="mt-3">{message.content || "Thinking..."}</MarkdownContent>
        {message.response?.citations.length ? (
          <details className="mt-4 rounded-2xl border border-border bg-card/60 p-3">
            <summary className="cursor-pointer text-sm font-medium text-foreground">
              Sources, conflicts, and retrieval trace
            </summary>
            <div className="mt-3 space-y-4">
              <div className="flex flex-wrap gap-2">
                {message.response.citations.map((citation, index) => (
                  <CitationChip key={`${citation.filename}-${index}`} citation={citation} index={index} />
                ))}
              </div>
              {message.response.conflicts.length ? (
                <div className="space-y-2">
                  {message.response.conflicts.map((conflict, index) => (
                    <div key={index} className="rounded-xl border border-warning/30 bg-warning/10 p-3 text-sm text-amber-100">
                      {conflict.explanation}
                    </div>
                  ))}
                </div>
              ) : null}
              <div className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-3">
                <TraceValue label="Mode" value={message.response.retrieval.mode} />
                <TraceValue label="Sparse" value={message.response.retrieval.sparse_available === false ? "unavailable" : "available"} />
                <TraceValue label="Top K" value={message.response.retrieval.top_k ?? "-"} />
              </div>
            </div>
          </details>
        ) : null}
      </div>
    </div>
  );
}

function toThreadMessage(message: ApiChatMessage): ThreadMessage {
  if (message.role === "user") {
    return {
      id: message.id,
      role: "user",
      content: message.content,
    };
  }

  return {
    id: message.id,
    role: "assistant",
    content: message.content,
    response: {
      id: message.result?.query_id ?? message.query_log_id ?? message.id,
      answer: message.content,
      citations: message.result?.citations ?? [],
      conflicts: [],
      retrieval: {
        mode: "dense_only",
        selected_chunks: message.result?.citations ?? [],
      },
      memory_hits: [],
      latency_ms: 0,
      reasoning_effort: "medium",
      created_at: message.created_at ?? new Date().toISOString(),
      session_id: message.session_id,
      raw: message,
    },
  };
}

function TraceValue({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-background p-2">
      <div>{label}</div>
      <div className="mt-1 font-mono text-foreground">{value}</div>
    </div>
  );
}
