"use client";

import { useMemo, useState } from "react";
import type { QueryHistoryItem } from "@/lib/api/schemas";
import { CitationChip } from "@/components/chat/citation-chip";
import { ConflictAlert } from "@/components/chat/conflict-alert";
import { RetrievalTraceView } from "@/components/chat/retrieval-trace";
import { JsonViewer } from "@/components/common/json-viewer";
import { MarkdownContent } from "@/components/common/markdown-content";
import { StatusBadge } from "@/components/common/status-badge";
import { formatDate, truncate } from "@/lib/utils";

export function QueryHistoryTable({ rows }: { rows: QueryHistoryItem[] }) {
  const [expanded, setExpanded] = useState("");
  const [search, setSearch] = useState("");
  const [conflictsOnly, setConflictsOnly] = useState(false);
  const [retrievalMode, setRetrievalMode] = useState("");
  const [effort, setEffort] = useState("");

  const filtered = useMemo(
    () =>
      rows.filter((row) => {
        if (search && !`${row.question} ${row.answer}`.toLowerCase().includes(search.toLowerCase())) return false;
        if (conflictsOnly && !row.conflicts.length) return false;
        if (retrievalMode && row.retrieval.mode !== retrievalMode) return false;
        if (effort && row.reasoning_effort !== effort) return false;
        return true;
      }),
    [conflictsOnly, effort, retrievalMode, rows, search],
  );

  return (
    <div className="space-y-3">
      <div className="grid gap-2 md:grid-cols-5">
        <input className="control md:col-span-2" placeholder="Search questions and answers" value={search} onChange={(event) => setSearch(event.target.value)} />
        <select className="control" value={retrievalMode} onChange={(event) => setRetrievalMode(event.target.value)}>
          <option value="">all retrieval</option>
          <option value="hybrid">hybrid</option>
          <option value="dense_only">dense_only</option>
          <option value="sparse_only">sparse_only</option>
        </select>
        <select className="control" value={effort} onChange={(event) => setEffort(event.target.value)}>
          <option value="">all effort</option>
          <option value="low">low</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
        </select>
        <label className="flex items-center gap-2 rounded-md border border-border bg-muted px-3 text-sm">
          <input type="checkbox" checked={conflictsOnly} onChange={(event) => setConflictsOnly(event.target.checked)} />
          Conflict only
        </label>
      </div>
      <div className="overflow-x-auto rounded-lg border border-border bg-card">
        <table className="w-full min-w-[980px] text-left text-sm">
          <thead className="border-b border-border text-xs text-muted-foreground">
            <tr>
              <th className="p-3 font-medium">Time</th>
              <th className="p-3 font-medium">Question</th>
              <th className="p-3 font-medium">Workspace</th>
              <th className="p-3 font-medium">Retrieval</th>
              <th className="p-3 font-medium">Citations</th>
              <th className="p-3 font-medium">Latency</th>
              <th className="p-3 font-medium">Conflicts</th>
              <th className="p-3 font-medium">Effort</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {filtered.map((row) => (
              <>
                <tr key={row.id} className="cursor-pointer hover:bg-muted/30" onClick={() => setExpanded(expanded === row.id ? "" : row.id)}>
                  <td className="p-3 font-mono text-xs text-muted-foreground">{formatDate(row.created_at)}</td>
                  <td className="max-w-sm p-3">{truncate(row.question, 92)}</td>
                  <td className="p-3 font-mono text-xs text-muted-foreground">{row.client_id}</td>
                  <td className="p-3"><StatusBadge status="neutral" label={row.retrieval.mode} /></td>
                  <td className="p-3 font-mono text-xs">{row.citations.length}</td>
                  <td className="p-3 font-mono text-xs">{row.latency_ms} ms</td>
                  <td className="p-3">{row.conflicts.length ? <StatusBadge status="warning" label={String(row.conflicts.length)} /> : <StatusBadge status="ok" label="0" />}</td>
                  <td className="p-3 font-mono text-xs">{row.reasoning_effort}</td>
                </tr>
                {expanded === row.id && (
                  <tr>
                    <td colSpan={8} className="bg-background p-4">
                      <div className="grid gap-4 lg:grid-cols-2">
                        <div className="space-y-3">
                          <MarkdownContent>{row.answer}</MarkdownContent>
                          <div className="flex flex-wrap gap-2">{row.citations.map((citation, index) => <CitationChip key={index} citation={citation} index={index} />)}</div>
                          {row.conflicts.map((conflict, index) => <ConflictAlert key={index} conflict={conflict} />)}
                        </div>
                        <div className="space-y-3">
                          <RetrievalTraceView retrieval={row.retrieval} />
                          <JsonViewer value={row} />
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
