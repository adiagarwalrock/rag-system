"use client";

import { StatusBadge } from "@/components/common/status-badge";
import type { QdrantNode } from "@/lib/api/schemas";
import { cn, truncate } from "@/lib/utils";

export const maxSelectedNodes = 4;

export function NodeSelectionTable({
  rows,
  selectedIds,
  onToggle,
}: {
  rows: QdrantNode[];
  selectedIds: string[];
  onToggle: (id: string) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-card">
      <table className="w-full min-w-[1120px] text-left text-sm">
        <thead className="border-b border-border text-xs text-muted-foreground">
          <tr>
            <th className="w-12 p-3 font-medium">Pick</th>
            <th className="p-3 font-medium">Node ID</th>
            <th className="p-3 font-medium">Client</th>
            <th className="p-3 font-medium">Document</th>
            <th className="p-3 font-medium">Parser</th>
            <th className="p-3 font-medium">Chunk</th>
            <th className="p-3 font-medium">Page</th>
            <th className="p-3 font-medium">Preview</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((node) => {
            const selected = selectedIds.includes(node.id);
            const disabled = !selected && selectedIds.length >= maxSelectedNodes;
            return (
              <tr key={node.id} className={cn("hover:bg-muted/30", selected && "bg-muted/20")}>
                <td className="p-3">
                  <input
                    type="checkbox"
                    checked={selected}
                    disabled={disabled}
                    onChange={() => onToggle(node.id)}
                    aria-label={`Select node ${node.id}`}
                  />
                </td>
                <td className="max-w-[190px] truncate p-3 font-mono text-xs">{node.id}</td>
                <td className="max-w-[180px] truncate p-3 font-mono text-xs text-muted-foreground">
                  {node.client_id ?? "-"}
                </td>
                <td className="max-w-[220px] truncate p-3">{node.document_name ?? node.document_id ?? "-"}</td>
                <td className="p-3">
                  <StatusBadge status="neutral" label={node.parser_name ?? "unknown"} />
                </td>
                <td className="max-w-[150px] truncate p-3 font-mono text-xs text-muted-foreground">
                  {node.chunk_type ?? "-"}
                </td>
                <td className="p-3 font-mono text-xs">{node.page_num ?? "-"}</td>
                <td className="max-w-md p-3 text-muted-foreground">{truncate(node.text_preview ?? "-", 140)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function selectedCountLabel(selectedCount: number) {
  return selectedCount >= 2 ? `${selectedCount} nodes selected` : `${selectedCount}/2 nodes selected`;
}
