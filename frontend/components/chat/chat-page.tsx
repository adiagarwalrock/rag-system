"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { usePathname, useRouter } from "next/navigation";
import { apiClient } from "@/lib/api/client";
import type { ChatMessage as ApiChatMessage, QueryRequest, QueryResponse } from "@/lib/api/schemas";
import { useSessionMessages } from "@/lib/hooks/use-chat";
import { useClients } from "@/lib/hooks/use-clients";
import { useWorkspaceStore } from "@/lib/state/workspace-store";
import { ChatComposer } from "@/components/chat/chat-composer";
import { Check, Copy, Printer, Share2 } from "lucide-react";
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
  phases?: string[];
  reasoning?: string;
};

export default function ChatPage({ routeSessionId }: { routeSessionId?: string }) {
  const router = useRouter();
  const pathname = usePathname();
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
  const [shareCopied, setShareCopied] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (routeSessionId && routeSessionId !== sessionId) {
      setSessionId(routeSessionId);
    }
  }, [routeSessionId, sessionId, setSessionId]);

  useEffect(() => {
    if (!workspaceId && clients.data?.[0]) setWorkspaceId(clients.data[0].id);
  }, [clients.data, setWorkspaceId, workspaceId]);

  useEffect(() => {
    if (!sessionId) {
      setMessages([]);
      return;
    }
    // Skip sync while streaming to avoid overwriting optimistic messages mid-flight.
    if (sessionMessages.data && !isStreaming) {
      setMessages(sessionMessages.data.map(toThreadMessage));
    }
  }, [sessionId, sessionMessages.data, isStreaming]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages, isStreaming]);

  function resolveSessionUrl(nextSessionId: string) {
    return `/c/${encodeURIComponent(nextSessionId)}`;
  }

  function setResolvedSession(nextSessionId?: string) {
    if (!nextSessionId) return;
    setSessionId(nextSessionId);
    queryClient.invalidateQueries({ queryKey: ["sessions", workspaceId] });
    queryClient.invalidateQueries({
      queryKey: ["session-messages", workspaceId, nextSessionId],
    });

    const nextPath = resolveSessionUrl(nextSessionId);
    if (pathname !== nextPath) router.replace(nextPath);
  }

  async function copyShareUrl() {
    if (!sessionId) return;
    const url = new URL(resolveSessionUrl(sessionId), window.location.origin);
    await navigator.clipboard.writeText(url.toString());
    setShareCopied(true);
    window.setTimeout(() => setShareCopied(false), 1200);
  }

  async function applyFallback(
    input: QueryRequest,
    assistantId: string,
    signal: AbortSignal,
  ) {
    const fallback = await apiClient.query(input, signal);
    setResolvedSession(fallback.session_id);
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
          onStatus: (phase) =>
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, phases: [...(message.phases ?? []), phase] }
                  : message,
              ),
            ),
          onReasoning: (delta) =>
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, reasoning: (message.reasoning ?? "") + delta }
                  : message,
              ),
            ),
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
            setResolvedSession(final.session_id);
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: final.answer,
                      response: final,
                      reasoning: (final as Record<string, unknown>).reasoning as string | undefined ?? message.reasoning,
                      streaming: false,
                      phases: undefined,
                    }
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
    <div className="print-chat relative flex h-full min-h-0 flex-col overflow-hidden">
      <div className="no-print pointer-events-none absolute right-3 top-3 z-10 flex gap-2 md:right-6 md:top-5">
        <button
          type="button"
          className="pointer-events-auto inline-flex h-9 items-center gap-2 rounded-full border border-border bg-card/90 px-3 text-xs font-medium text-muted-foreground shadow-lg shadow-black/20 transition hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
          disabled={!messages.length}
          onClick={() => window.print()}
          aria-label="Print chat"
        >
          <Printer className="h-4 w-4" />
          Print
        </button>
        <button
          type="button"
          className="pointer-events-auto inline-flex h-9 items-center gap-2 rounded-full border border-border bg-card/90 px-3 text-xs font-medium text-muted-foreground shadow-lg shadow-black/20 transition hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
          disabled={!sessionId}
          onClick={copyShareUrl}
          aria-label="Copy chat URL"
        >
          {shareCopied ? <Check className="h-4 w-4 text-green-300" /> : <Share2 className="h-4 w-4" />}
          {shareCopied ? "Copied" : "Share"}
        </button>
      </div>
      <div className="no-scrollbar min-h-0 flex-1 overflow-y-auto px-3 pb-80 pt-10 md:pt-16">
        <div className="chat-print-content mx-auto flex min-h-full w-full max-w-3xl flex-col">
          {messages.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center pb-32 text-center">
              <h1 className="text-2xl font-medium text-foreground">Good to see you.</h1>
              <p className="mt-3 max-w-xl text-sm leading-6 text-muted-foreground">
                Ask across documents in the selected client workspace. Use the plus button in the composer to switch clients.
              </p>
              <button
                className="mt-6 rounded-full border border-border bg-card px-4 py-2 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
                onClick={() => submit("What are the key findings in these documents?")}
                disabled={!workspaceId || isStreaming}
              >
                What are the key findings in these documents?
              </button>
            </div>
          ) : (
            <div className="chat-print-thread flex flex-col gap-7">
              {messages.map((message) => (
                <ChatThreadMessage key={message.id} message={message} />
              ))}
              <div ref={bottomRef} className="h-10" />
            </div>
          )}
        </div>
      </div>

      <div className="no-print pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-background via-background to-transparent px-3 pb-2 pt-12">
        <div className="pointer-events-auto mx-auto max-w-3xl">
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
          <p className="mt-1 text-center text-[11px] text-muted-foreground">
            Development preview. Verify answers against cited source material before relying on them.
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
      <div className="chat-print-turn chat-print-user flex justify-end">
        <div className="chat-print-bubble max-w-[82%] rounded-2xl rounded-br-md bg-blue-600 px-4 py-3 text-sm leading-6 text-white">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="chat-print-turn chat-print-assistant w-full">
      <div className="min-w-0">
        {/* Live phase status panel — shown while streaming, cleared on final */}
        {message.streaming && message.phases && message.phases.length > 0 ? (
          <div className="mt-3 rounded-xl border border-border bg-muted/20 px-4 py-3 text-xs text-muted-foreground space-y-1.5">
            <div className="font-medium text-foreground/60 mb-2">Processing your question…</div>
            {message.phases.map((phase, i) => (
              <div key={i}>{phase}</div>
            ))}
          </div>
        ) : null}

        {/* Reasoning summary — shown while streaming (live) and after response (collapsed) */}
        {message.reasoning ? (
          <details className="mt-3 rounded-xl border border-border bg-muted/20" open={message.streaming}>
            <summary className="flex cursor-pointer items-center gap-2 px-4 py-2.5 text-xs font-medium text-muted-foreground">
              <span>🧠</span> Model reasoning summary
            </summary>
            <MarkdownContent className="px-4 pb-3 pt-1 text-xs leading-5 text-foreground/80">
              {message.reasoning}
            </MarkdownContent>
          </details>
        ) : null}

        <MarkdownContent className="mt-3">{message.content || (message.streaming && !message.phases?.length ? "Thinking..." : "")}</MarkdownContent>
        {message.response?.citations.length ? (
          <details className="no-print mt-4 rounded-2xl border border-border bg-card/60 p-3">
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
            </div>
          </details>
        ) : null}
        {message.response?.citations.length ? (
          <div className="print-sources">
            <div className="print-section-title">Sources</div>
            <ol>
              {message.response.citations.map((citation, index) => (
                <li key={`${citation.filename}-${index}`}>
                  <span>{citation.filename}</span>
                  {citation.page ? <span> page {citation.page}</span> : null}
                  {citation.score !== undefined ? <span> score {citation.score.toFixed(3)}</span> : null}
                  {citation.quote ? <blockquote>{citation.quote}</blockquote> : null}
                </li>
              ))}
            </ol>
          </div>
        ) : null}
        {message.response ? (
          <div className="no-print mt-4 flex flex-wrap items-center gap-2">
            {message.response.conflicts.length ? <StatusBadge status="warning" label={`${message.response.conflicts.length} conflicts`} /> : null}
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
        mode:
          (message as Record<string, unknown>).retrieval_mode === "hybrid"
            ? "hybrid"
            : "dense_only",
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
