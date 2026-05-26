"use client";

import { Copy, RotateCcw } from "lucide-react";
import type { QueryResponse } from "@/lib/api/schemas";
import { CitationChip } from "@/components/chat/citation-chip";
import { StatusBadge } from "@/components/common/status-badge";

export function AnswerCard({
  response,
  streamingText,
  isStreaming,
  onRegenerate,
}: {
  response?: QueryResponse;
  streamingText?: string;
  isStreaming?: boolean;
  onRegenerate?: () => void;
}) {
  const answer = response?.answer ?? streamingText ?? "";
  if (!answer && !isStreaming) return null;

  return (
    <article className="rounded-lg border border-border bg-card">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3">
        <div className="flex flex-wrap gap-2">
          <StatusBadge status={isStreaming ? "processing" : "ok"} label={isStreaming ? "streaming" : "answered"} />
          {response && <StatusBadge status="neutral" label={response.retrieval.mode} />}
          {response?.retrieval.sparse_available === false && <StatusBadge status="warning" label="dense-only fallback" />}
          {response && <StatusBadge status="neutral" label={`${response.latency_ms} ms`} />}
          {response && <StatusBadge status="neutral" label={`reasoning ${response.reasoning_effort}`} />}
          {response?.conflicts.length ? <StatusBadge status="warning" label={`${response.conflicts.length} conflicts`} /> : null}
        </div>
        <div className="flex gap-1">
          <button className="button-ghost h-8 w-8 p-0" aria-label="Copy answer" onClick={() => navigator.clipboard.writeText(answer)}>
            <Copy className="h-4 w-4" />
          </button>
          <button className="button-ghost h-8 w-8 p-0" aria-label="Regenerate" onClick={onRegenerate} disabled={!onRegenerate}>
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>
      </div>
      <div className="space-y-4 p-4">
        <p className="whitespace-pre-wrap text-sm leading-6 text-foreground">{answer}</p>
        {response?.citations.length ? (
          <div className="flex flex-wrap gap-2">
            {response.citations.map((citation, index) => (
              <CitationChip key={`${citation.filename}-${index}`} citation={citation} index={index} />
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}
