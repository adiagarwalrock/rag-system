"use client";

import { Fragment, useMemo, useState } from "react";
import { Check, Copy, FileImage, Trash2 } from "lucide-react";
import type { Citation, QueryHistoryItem } from "@/lib/api/schemas";
import { ConflictAlert } from "@/components/chat/conflict-alert";
import { RetrievalTraceView } from "@/components/chat/retrieval-trace";
import { ConfirmDeleteDialog } from "@/components/common/confirm-delete-dialog";
import { JsonViewer } from "@/components/common/json-viewer";
import { MarkdownContent } from "@/components/common/markdown-content";
import { StatusBadge } from "@/components/common/status-badge";
import { DarkSelect } from "@/components/common/dark-select";
import { formatDate, truncate } from "@/lib/utils";

export function QueryHistoryTable({
  rows,
  deleting,
  onDelete,
}: {
  rows: QueryHistoryItem[];
  deleting?: boolean;
  onDelete?: (queryId: string) => Promise<unknown>;
}) {
  const [expanded, setExpanded] = useState("");
  const [copied, setCopied] = useState("");
  const [search, setSearch] = useState("");
  const [conflictsOnly, setConflictsOnly] = useState(false);
  const [effort, setEffort] = useState("");

  const filtered = useMemo(
    () =>
      rows.filter((row) => {
        if (search && !`${row.question} ${row.answer}`.toLowerCase().includes(search.toLowerCase())) return false;
        if (conflictsOnly && !row.conflicts.length) return false;
        if (effort && row.reasoning_effort !== effort) return false;
        return true;
      }),
    [conflictsOnly, effort, rows, search],
  );

  async function copyToClipboard(key: string, value: string) {
    await navigator.clipboard.writeText(value);
    setCopied(key);
    window.setTimeout(() => {
      setCopied((current) => (current === key ? "" : current));
    }, 1200);
  }

  return (
    <div className="space-y-3">
      <div className="grid gap-2 md:grid-cols-4">
        <input className="control md:col-span-2" placeholder="Search questions and answers" value={search} onChange={(event) => setSearch(event.target.value)} />
        <DarkSelect
          label="Reasoning effort filter"
          value={effort}
          placeholder="all effort"
          onChange={setEffort}
          options={[
            { value: "", label: "all effort" },
            { value: "low", label: "low" },
            { value: "medium", label: "medium" },
            { value: "high", label: "high" },
          ]}
        />
        <label className="flex items-center gap-2 rounded-md border border-border bg-muted px-3 text-sm">
          <input type="checkbox" checked={conflictsOnly} onChange={(event) => setConflictsOnly(event.target.checked)} />
          Conflict only
        </label>
      </div>
      <div className="overflow-x-auto rounded-lg border border-border bg-card">
        <table className="w-full min-w-[760px] text-left text-sm">
          <thead className="border-b border-border text-xs text-muted-foreground">
            <tr>
              <th className="p-3 font-medium">Time</th>
              <th className="p-3 font-medium">Question</th>
              <th className="p-3 font-medium">Citations</th>
              <th className="p-3 font-medium">Latency</th>
              <th className="p-3 font-medium">Conflicts</th>
              <th className="p-3 font-medium">Effort</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {filtered.map((row) => (
              <Fragment key={row.id}>
                <tr className="cursor-pointer hover:bg-muted/30" onClick={() => setExpanded(expanded === row.id ? "" : row.id)}>
                  <td className="p-3 font-mono text-xs text-muted-foreground">{formatDate(row.created_at)}</td>
                  <td className="max-w-sm p-3">{truncate(row.question, 92)}</td>
                  <td className="p-3 font-mono text-xs">{row.citations.length}</td>
                  <td className="p-3 font-mono text-xs">{formatLatency(row.latency_ms)}</td>
                  <td className="p-3">{row.conflicts.length ? <StatusBadge status="warning" label={String(row.conflicts.length)} /> : <StatusBadge status="ok" label="0" />}</td>
                  <td className="p-3 font-mono text-xs">{row.reasoning_effort}</td>
                </tr>
                {expanded === row.id && (
                  <tr>
                    <td colSpan={6} className="bg-background p-4">
                      <div className="grid gap-4 lg:grid-cols-2">
                        <div className="space-y-3">
                          <div className="flex flex-wrap items-center gap-2">
                            <CopyButton
                              copied={copied === `${row.id}:response`}
                              label="Copy raw response"
                              onClick={() => copyToClipboard(`${row.id}:response`, row.answer)}
                            />
                            <CopyButton
                              copied={copied === `${row.id}:json`}
                              label="Copy raw JSON"
                              onClick={() =>
                                copyToClipboard(
                                  `${row.id}:json`,
                                  JSON.stringify(row.raw ?? row, null, 2),
                                )
                              }
                            />
                            {onDelete ? (
                              <ConfirmDeleteDialog
                                title="Delete history item"
                                description="This deletes the query history row, retrieval logs, and conflict logs. The chat transcript remains intact."
                                pending={deleting}
                                onConfirm={async () => {
                                  await onDelete(row.id);
                                  setExpanded("");
                                }}
                              >
                                <button
                                  type="button"
                                  className="inline-flex h-8 items-center gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-2.5 text-xs text-red-200 transition hover:bg-destructive/20"
                                  disabled={deleting}
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                  Delete
                                </button>
                              </ConfirmDeleteDialog>
                            ) : null}
                          </div>
                          <MarkdownContent>{row.answer}</MarkdownContent>
                          <HistoryCitationList citations={row.citations} />
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
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CopyButton({
  copied,
  label,
  onClick,
}: {
  copied: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="inline-flex h-8 items-center gap-2 rounded-md border border-border bg-muted px-2.5 text-xs text-muted-foreground transition hover:bg-muted/70 hover:text-foreground"
      onClick={onClick}
    >
      {copied ? <Check className="h-3.5 w-3.5 text-green-300" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? "Copied" : label}
    </button>
  );
}

function formatLatency(value?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  if (value >= 1000) return `${(value / 1000).toFixed(value >= 10_000 ? 1 : 2)} s`;
  return `${Math.round(value)} ms`;
}

function HistoryCitationList({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;

  return (
    <div className="space-y-2">
      {citations.map((citation, index) => {
        const extras = citation as Record<string, unknown>;
        const documentName = String(extras.document_name ?? citation.filename);
        return (
          <div
            key={`${citation.document_id ?? citation.filename}-${citation.chunk_id ?? index}`}
            className="rounded-md border border-border bg-muted/40 px-3 py-2 text-xs"
          >
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
              <span className="font-mono text-muted-foreground">[{index + 1}]</span>
              <span className="font-medium text-foreground">{documentName}</span>
              {citation.page ? (
                <span className="font-mono text-blue-200">page {citation.page}</span>
              ) : null}
            </div>
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[11px] text-muted-foreground">
              <span>document_id: {citation.document_id ?? "-"}</span>
              <span>chunk_id: {citation.chunk_id ?? "-"}</span>
              {extras.source_artifact_id ? <span>artifact: {String(extras.source_artifact_id)}</span> : null}
              {extras.source_artifact_type ? <span>type: {String(extras.source_artifact_type)}</span> : null}
              {documentName !== citation.filename ? <span>file: {citation.filename}</span> : null}
            </div>
            {citation.image_assets.length ? (
              <div className="mt-2 grid grid-cols-3 gap-2">
                {citation.image_assets.map((asset) => (
                  <a key={`${asset.url}-${asset.filename}`} href={asset.url} target="_blank" rel="noreferrer" className="overflow-hidden rounded-md border border-border bg-background">
                    <div className="aspect-[4/3]">
                      <img src={asset.url} alt={asset.filename} className="h-full w-full object-contain" loading="lazy" />
                    </div>
                    <div className="flex items-center gap-1 border-t border-border px-1.5 py-1 text-[10px] text-muted-foreground">
                      <FileImage className="h-3 w-3 shrink-0" />
                      <span className="truncate">{asset.filename}</span>
                    </div>
                  </a>
                ))}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
