"use client";

import { JsonViewer } from "@/components/common/json-viewer";
import { MarkdownContent } from "@/components/common/markdown-content";
import { StatusBadge } from "@/components/common/status-badge";
import type { QdrantNode } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";

const compareFields = [
  ["document_name", "Document"],
  ["client_id", "Client"],
  ["parser_name", "Parser"],
  ["parser_version", "Parser version"],
  ["chunk_type", "Chunk type"],
  ["page_num", "Page"],
  ["section_path", "Section"],
] as const;

export function NodeComparison({ nodes }: { nodes: QdrantNode[] }) {
  const differingFields = new Set<string>();
  for (const [field] of compareFields) {
    const values = new Set(nodes.map((node) => String(node[field] ?? "")));
    if (values.size > 1) differingFields.add(field);
  }

  return (
    <div className="overflow-x-auto">
      <div
        className="grid gap-4"
        style={{
          gridTemplateColumns: `repeat(${nodes.length}, minmax(340px, 1fr))`,
          minWidth: `${nodes.length * 360}px`,
        }}
      >
        {nodes.map((node, index) => (
          <article key={node.id} className="rounded-lg border border-border bg-card">
            <div className="border-b border-border p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs font-medium text-muted-foreground">
                    Node {index + 1}
                  </div>
                  <h2 className="mt-1 truncate text-sm font-semibold">
                    {node.document_name ?? node.id}
                  </h2>
                </div>
                <StatusBadge status="neutral" label={node.parser_name ?? "unknown"} />
              </div>
              <div className="mt-3 space-y-1 font-mono text-[11px] text-muted-foreground">
                <div className="truncate">{node.id}</div>
                <div className="truncate">client: {node.client_id ?? "-"}</div>
                <div className="truncate">{node.citation_label ?? "No citation label"}</div>
              </div>
            </div>
            <div className="space-y-4 p-4">
              <div className="grid grid-cols-2 gap-2">
                {compareFields.map(([field, label]) => (
                  <MetadataPill
                    key={field}
                    label={label}
                    value={formatNodeValue(node[field])}
                    differing={differingFields.has(field)}
                  />
                ))}
              </div>
              <div>
                <h3 className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
                  Parsed text
                </h3>
                <div className="max-h-[520px] overflow-auto rounded-md border border-border bg-background p-3">
                  {node.text ? (
                    <MarkdownContent className="text-sm leading-6">{node.text}</MarkdownContent>
                  ) : (
                    <div className="text-sm text-muted-foreground">
                      No parsed text found for this node.
                    </div>
                  )}
                </div>
              </div>
              <div>
                <h3 className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
                  Node metadata
                </h3>
                <JsonViewer value={node.node_metadata} wrap />
              </div>
              <details className="rounded-md border border-border bg-background">
                <summary className="cursor-pointer px-3 py-2 text-xs font-semibold uppercase text-muted-foreground">
                  Top-level metadata
                </summary>
                <div className="border-t border-border p-3">
                  <JsonViewer value={node.top_level_metadata} wrap />
                </div>
              </details>
              <details className="rounded-md border border-border bg-background">
                <summary className="cursor-pointer px-3 py-2 text-xs font-semibold uppercase text-muted-foreground">
                  Raw payload
                </summary>
                <div className="border-t border-border p-3">
                  <JsonViewer value={node.raw_payload ?? {}} wrap />
                </div>
              </details>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

function MetadataPill({
  label,
  value,
  differing,
}: {
  label: string;
  value: string;
  differing: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-md border p-2",
        differing ? "border-warning/40 bg-warning/10" : "border-border bg-background",
      )}
    >
      <div className="text-[10px] font-medium uppercase text-muted-foreground">{label}</div>
      <div className="mt-1 min-h-5 break-words font-mono text-xs">{value || "-"}</div>
    </div>
  );
}

function formatNodeValue(value: unknown) {
  if (value === null || value === undefined || value === "") return "-";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}
