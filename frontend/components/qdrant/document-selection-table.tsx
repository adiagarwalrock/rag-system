"use client";

import { StatusBadge } from "@/components/common/status-badge";
import type { QdrantDocumentCandidate } from "@/lib/api/schemas";
import { cn, truncate } from "@/lib/utils";

export const maxSelectedDocuments = 4;

export function DocumentSelectionTable({
  rows,
  selectedKeys,
  onToggle,
}: {
  rows: QdrantDocumentCandidate[];
  selectedKeys: string[];
  onToggle: (documentKey: string) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-card">
      <table className="w-full min-w-[1040px] text-left text-sm">
        <thead className="border-b border-border text-xs text-muted-foreground">
          <tr>
            <th className="w-12 p-3 font-medium">Pick</th>
            <th className="p-3 font-medium">Client</th>
            <th className="p-3 font-medium">Document</th>
            <th className="p-3 font-medium">Parser</th>
            <th className="p-3 font-medium">Version</th>
            <th className="p-3 font-medium">Nodes</th>
            <th className="p-3 font-medium">Pages</th>
            <th className="p-3 font-medium">Preview</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((document) => {
            const selected = selectedKeys.includes(document.document_key);
            const disabled = !selected && selectedKeys.length >= maxSelectedDocuments;
            return (
              <tr
                key={document.document_key}
                className={cn("hover:bg-muted/30", selected && "bg-muted/20")}
              >
                <td className="p-3">
                  <input
                    type="checkbox"
                    checked={selected}
                    disabled={disabled}
                    onChange={() => onToggle(document.document_key)}
                    aria-label={`Select document ${document.document_name ?? document.document_id ?? document.document_key}`}
                  />
                </td>
                <td className="max-w-[180px] truncate p-3 font-mono text-xs text-muted-foreground">
                  {document.client_id ?? "-"}
                </td>
                <td className="max-w-[260px] truncate p-3">
                  {document.document_name ?? document.document_id ?? "-"}
                </td>
                <td className="p-3">
                  <StatusBadge status="neutral" label={document.parser_name ?? "unknown"} />
                </td>
                <td className="max-w-[120px] truncate p-3 font-mono text-xs text-muted-foreground">
                  {document.parser_version ?? "-"}
                </td>
                <td className="p-3 font-mono text-xs">{document.node_count}</td>
                <td className="p-3 font-mono text-xs">{document.page_count}</td>
                <td className="max-w-md p-3 text-muted-foreground">
                  {truncate(document.preview ?? "-", 140)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function selectedDocumentCountLabel(selectedCount: number) {
  return selectedCount >= 2
    ? `${selectedCount} documents selected`
    : `${selectedCount}/2 documents selected`;
}
