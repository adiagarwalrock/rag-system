"use client";

import * as Tabs from "@radix-ui/react-tabs";
import type { QueryResponse } from "@/lib/api/schemas";
import { ConflictAlert } from "@/components/chat/conflict-alert";
import { RetrievalTraceView } from "@/components/chat/retrieval-trace";
import { JsonViewer } from "@/components/common/json-viewer";
import { scoreLabel } from "@/lib/utils";

const tabs = ["Sources", "Conflicts", "Retrieval", "Memory", "JSON"];

export function AnswerInspector({ response }: { response?: QueryResponse }) {
  if (!response) {
    return (
      <div className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground">
        Ask a question to inspect sources, conflicts, retrieval, memory, and raw JSON.
      </div>
    );
  }

  return (
    <Tabs.Root defaultValue="Sources" className="rounded-lg border border-border bg-card">
      <Tabs.List className="flex overflow-x-auto border-b border-border px-2">
        {tabs.map((tab) => (
          <Tabs.Trigger
            key={tab}
            value={tab}
            className="border-b-2 border-transparent px-3 py-2 text-xs text-muted-foreground data-[state=active]:border-primary data-[state=active]:text-foreground"
          >
            {tab}
          </Tabs.Trigger>
        ))}
      </Tabs.List>
      <div className="p-3">
        <Tabs.Content value="Sources" className="space-y-3">
          {response.citations.length ? response.citations.map((citation, index) => (
            <div key={`${citation.filename}-${index}`} className="rounded-md border border-border bg-muted/30 p-3">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="font-medium text-foreground">{citation.filename}</span>
                {citation.page && <span className="font-mono text-muted-foreground">page {citation.page}</span>}
                {citation.chunk_id && <span className="font-mono text-muted-foreground">{citation.chunk_id}</span>}
                {citation.score !== undefined && <span className="font-mono text-blue-300">score {scoreLabel(citation.score)}</span>}
              </div>
              {citation.quote && <p className="mt-2 text-sm text-muted-foreground">{citation.quote}</p>}
            </div>
          )) : <p className="text-sm text-muted-foreground">No citations returned.</p>}
        </Tabs.Content>
        <Tabs.Content value="Conflicts" className="space-y-3">
          {response.conflicts.length ? response.conflicts.map((conflict, index) => (
            <ConflictAlert key={index} conflict={conflict} />
          )) : <p className="text-sm text-muted-foreground">No conflicts detected.</p>}
        </Tabs.Content>
        <Tabs.Content value="Retrieval">
          <RetrievalTraceView retrieval={response.retrieval} />
        </Tabs.Content>
        <Tabs.Content value="Memory" className="space-y-3">
          {response.memory_hits.length ? response.memory_hits.map((hit) => (
            <div key={hit.id} className="rounded-md border border-border bg-muted/30 p-3">
              <div className="font-mono text-xs text-blue-300">{scoreLabel(hit.score)} {hit.session_id}</div>
              <p className="mt-2 text-sm text-foreground">{hit.question}</p>
              <p className="mt-1 text-sm text-muted-foreground">{hit.answer_preview}</p>
            </div>
          )) : <p className="text-sm text-muted-foreground">No memory hits used.</p>}
        </Tabs.Content>
        <Tabs.Content value="JSON">
          <JsonViewer value={response.raw ?? response} />
        </Tabs.Content>
      </div>
    </Tabs.Root>
  );
}
