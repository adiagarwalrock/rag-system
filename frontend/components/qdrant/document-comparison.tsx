"use client";

import { JsonViewer } from "@/components/common/json-viewer";
import { MarkdownContent } from "@/components/common/markdown-content";
import { StatusBadge } from "@/components/common/status-badge";
import type { QdrantDocumentCompareItem } from "@/lib/api/schemas";
import { cn, truncate } from "@/lib/utils";

const compareFields = [
  ["document_name", "Document"],
  ["client_id", "Client"],
  ["parser_name", "Parser"],
  ["parser_version", "Parser version"],
  ["node_count", "Nodes"],
  ["page_count", "Pages"],
] as const;

export function DocumentComparison({
  documents,
}: {
  documents: QdrantDocumentCompareItem[];
}) {
  const differingFields = new Set<string>();
  for (const [field] of compareFields) {
    const values = new Set(documents.map((document) => String(document[field] ?? "")));
    if (values.size > 1) differingFields.add(field);
  }

  return (
    <div className="overflow-x-auto">
      <div
        className="grid gap-4"
        style={{
          gridTemplateColumns: `repeat(${documents.length}, minmax(360px, 1fr))`,
          minWidth: `${documents.length * 380}px`,
        }}
      >
        {documents.map((document, index) => (
          <article key={document.document_key} className="rounded-lg border border-border bg-card">
            <div className="border-b border-border p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs font-medium text-muted-foreground">
                    Document {index + 1}
                  </div>
                  <h2 className="mt-1 truncate text-sm font-semibold">
                    {document.document_name ?? document.document_id ?? "Indexed document"}
                  </h2>
                </div>
                <StatusBadge status="neutral" label={document.parser_name ?? "unknown"} />
              </div>
              <div className="mt-3 space-y-1 font-mono text-[11px] text-muted-foreground">
                <div className="truncate">client: {document.client_id ?? "-"}</div>
                <div className="truncate">document: {document.document_id ?? "-"}</div>
                <div className="truncate">key: {document.document_key}</div>
              </div>
            </div>
            <div className="space-y-4 p-4">
              <div className="grid grid-cols-2 gap-2">
                {compareFields.map(([field, label]) => (
                  <MetadataPill
                    key={field}
                    label={label}
                    value={formatValue(document[field])}
                    differing={differingFields.has(field)}
                  />
                ))}
              </div>
              <div>
                <h3 className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
                  Reconstructed markdown
                </h3>
                <div className="max-h-[640px] overflow-auto rounded-md border border-border bg-background p-3">
                  {document.markdown ? (
                    <MarkdownContent className="text-sm leading-6">
                      {document.markdown}
                    </MarkdownContent>
                  ) : (
                    <div className="text-sm text-muted-foreground">
                      No parsed markdown found for this document.
                    </div>
                  )}
                </div>
              </div>
              <details className="rounded-md border border-border bg-background">
                <summary className="cursor-pointer px-3 py-2 text-xs font-semibold uppercase text-muted-foreground">
                  Metadata summary
                </summary>
                <div className="border-t border-border p-3">
                  <JsonViewer value={document.metadata_summary} wrap />
                </div>
              </details>
              <details className="rounded-md border border-border bg-background">
                <summary className="cursor-pointer px-3 py-2 text-xs font-semibold uppercase text-muted-foreground">
                  Source nodes
                </summary>
                <div className="max-h-96 overflow-auto border-t border-border">
                  <table className="w-full min-w-[560px] text-left text-xs">
                    <thead className="border-b border-border text-muted-foreground">
                      <tr>
                        <th className="p-2 font-medium">Node</th>
                        <th className="p-2 font-medium">Chunk</th>
                        <th className="p-2 font-medium">Page</th>
                        <th className="p-2 font-medium">Preview</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {document.source_nodes.map((node, nodeIndex) => (
                        <tr key={`${node.id ?? "node"}-${nodeIndex}`}>
                          <td className="max-w-[160px] truncate p-2 font-mono">
                            {node.id ?? "-"}
                          </td>
                          <td className="max-w-[120px] truncate p-2 font-mono text-muted-foreground">
                            {node.chunk_type ?? node.chunk_id ?? "-"}
                          </td>
                          <td className="p-2 font-mono">
                            {node.page_num ?? node.page_nums.join(", ") ?? "-"}
                          </td>
                          <td className="max-w-xs p-2 text-muted-foreground">
                            {truncate(node.text_preview ?? "-", 100)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
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

function formatValue(value: unknown) {
  if (value === null || value === undefined || value === "") return "-";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}
