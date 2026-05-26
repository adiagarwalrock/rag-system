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
import { BookOpen, Check, Copy, Printer, Share2, X } from "lucide-react";
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

const inspectorWidthKey = "rag-console.sources-panel-width";
const defaultInspectorWidth = 420;
const minInspectorWidth = 320;
const maxInspectorWidth = 720;

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
  const [inspectedResponse, setInspectedResponse] = useState<QueryResponse | null>(null);
  const [inspectorWidth, setInspectorWidth] = useState(defaultInspectorWidth);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const stored = Number(localStorage.getItem(inspectorWidthKey));
    if (Number.isFinite(stored)) setInspectorWidth(clampInspectorWidth(stored));
  }, []);

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

  function updateInspectorWidth(width: number) {
    const nextWidth = clampInspectorWidth(width);
    setInspectorWidth(nextWidth);
    localStorage.setItem(inspectorWidthKey, String(nextWidth));
  }

  function startInspectorResize(event: React.PointerEvent) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = inspectorWidth;

    function handlePointerMove(moveEvent: PointerEvent) {
      updateInspectorWidth(startWidth + startX - moveEvent.clientX);
    }

    function handlePointerUp() {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    }

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
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
                <ChatThreadMessage
                  key={message.id}
                  message={message}
                  onInspectSources={setInspectedResponse}
                />
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
      <AnswerSourcesPanel
        response={inspectedResponse}
        onClose={() => setInspectedResponse(null)}
        width={inspectorWidth}
        onResizeStart={startInspectorResize}
        onWidthChange={updateInspectorWidth}
      />
    </div>
  );
}

function ChatThreadMessage({
  message,
  onInspectSources,
}: {
  message: ThreadMessage;
  onInspectSources: (response: QueryResponse) => void;
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
            <button
              className="inline-flex h-6 items-center gap-1 rounded-md border border-border bg-muted px-2 text-xs text-muted-foreground hover:text-foreground"
              onClick={() => onInspectSources(message.response!)}
            >
              <BookOpen className="h-3.5 w-3.5" />
              Sources
            </button>
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

function AnswerSourcesPanel({
  response,
  onClose,
  width,
  onResizeStart,
  onWidthChange,
}: {
  response: QueryResponse | null;
  onClose: () => void;
  width: number;
  onResizeStart: (event: React.PointerEvent) => void;
  onWidthChange: (width: number) => void;
}) {
  if (!response) return null;

  return (
    <aside
      className="no-print fixed inset-y-0 right-0 z-40 flex flex-col border-l border-border bg-card shadow-2xl shadow-black/50"
      style={{ width: `min(${width}px, 92vw)` }}
    >
      <div
        role="separator"
        aria-label="Resize sources panel"
        aria-orientation="vertical"
        tabIndex={0}
        className="absolute inset-y-0 -left-1 flex w-2 cursor-col-resize touch-none items-center justify-center"
        onPointerDown={onResizeStart}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") {
            event.preventDefault();
            onWidthChange(width + 16);
          }
          if (event.key === "ArrowRight") {
            event.preventDefault();
            onWidthChange(width - 16);
          }
        }}
      >
        <span className="h-10 w-px rounded-full bg-border transition hover:bg-primary" />
      </div>
      <div className="flex h-14 items-center justify-between border-b border-border px-4">
        <div>
          <div className="text-sm font-semibold">Sources</div>
          <div className="text-xs text-muted-foreground">
            Evidence, conflicts, and retrieval trace
          </div>
        </div>
        <button
          type="button"
          className="button-ghost h-8 w-8 p-0"
          onClick={onClose}
          aria-label="Close sources"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      <div className="no-scrollbar min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <InspectorSection label={`Citations // ${response.citations.length}`}>
          {response.citations.length ? (
            <div className="space-y-2">
              {response.citations.map((citation, index) => (
                <div key={`${citation.filename}-${index}`} className="rounded-lg border border-border bg-background p-3">
                  <CitationChip citation={citation} index={index} />
                  {citation.quote ? (
                    <MarkdownContent className="mt-2 text-sm leading-6 text-muted-foreground">
                      {citation.quote}
                    </MarkdownContent>
                  ) : null}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No citations returned.</p>
          )}
        </InspectorSection>

        <InspectorSection label={`Conflicts // ${response.conflicts.length}`}>
          {response.conflicts.length ? (
            <div className="space-y-2">
              {response.conflicts.map((conflict, index) => (
                <div key={index} className="rounded-lg border border-warning/30 bg-warning/10 p-3 text-sm text-amber-100">
                  <div className="mb-1 font-medium">{conflict.severity} {conflict.type}</div>
                  <MarkdownContent className="text-sm leading-6 text-amber-100">
                    {conflict.explanation}
                  </MarkdownContent>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No conflicts detected.</p>
          )}
        </InspectorSection>

        <InspectorSection label="Retrieval">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <TraceFact label="Mode" value={response.retrieval.mode} />
            <TraceFact label="Sparse" value={response.retrieval.sparse_available === false ? "unavailable" : "available"} />
            <TraceFact label="Top K" value={response.retrieval.top_k ?? "-"} />
            <TraceFact label="Latency" value={`${response.latency_ms} ms`} />
          </div>
          {response.retrieval.fallback_reason ? (
            <div className="mt-2 rounded-lg border border-border bg-background p-3">
              <MarkdownContent className="text-xs leading-5 text-muted-foreground">
                {response.retrieval.fallback_reason}
              </MarkdownContent>
            </div>
          ) : null}
        </InspectorSection>
      </div>
    </aside>
  );
}

function InspectorSection({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <details className="group mb-3 rounded-lg border border-border bg-card/60">
      <summary className="flex cursor-pointer list-none items-center justify-between px-3 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground">
        <span>{label}</span>
        <span className="text-[11px] transition group-open:rotate-90">›</span>
      </summary>
      <div className="border-t border-border p-3">{children}</div>
    </details>
  );
}

function TraceFact({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-background p-3">
      <div className="text-muted-foreground">{label}</div>
      <div className="mt-1 font-mono text-foreground">{value}</div>
    </div>
  );
}

function clampInspectorWidth(width: number) {
  return Math.min(maxInspectorWidth, Math.max(minInspectorWidth, Math.round(width)));
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
