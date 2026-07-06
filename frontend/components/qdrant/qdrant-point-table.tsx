"use client";

import { Database } from "lucide-react";
import type { QdrantPoint } from "@/lib/api/schemas";
import { StatusBadge } from "@/components/common/status-badge";
import { truncate } from "@/lib/utils";

export function QdrantPointTable({
  points,
  onSelect,
}: {
  points: QdrantPoint[];
  onSelect: (point: QdrantPoint) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-card">
      <table className="w-full min-w-[1100px] text-left text-sm">
        <thead className="border-b border-border text-xs text-muted-foreground">
          <tr>
            <th className="p-3 font-medium">Point ID</th>
            <th className="p-3 font-medium">Document</th>
            <th className="p-3 font-medium">Page</th>
            <th className="p-3 font-medium">Chunk preview</th>
            <th className="p-3 font-medium">Dense size</th>
            <th className="p-3 font-medium">Sparse</th>
            <th className="p-3 font-medium">Family</th>
            <th className="p-3 font-medium">Version</th>
            <th className="p-3 font-medium">Metadata</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {points.map((point) => (
            <tr key={point.id} className="cursor-pointer hover:bg-muted/30" onClick={() => onSelect(point)}>
              <td className="max-w-[220px] truncate p-3 font-mono text-xs">{point.id}</td>
              <td className="max-w-[220px] truncate p-3">{point.filename ?? point.document_id ?? "-"}</td>
              <td className="p-3 font-mono text-xs">{point.page ?? "-"}</td>
              <td className="max-w-sm p-3 text-muted-foreground">{truncate(point.chunk_preview ?? "-", 120)}</td>
              <td className="p-3 font-mono text-xs">{point.dense_vector_size ?? "-"}</td>
              <td className="p-3"><StatusBadge status={point.sparse_vector_available ? "ok" : "degraded"} label={point.sparse_vector_available ? "yes" : "no"} /></td>
              <td className="max-w-[180px] truncate p-3 font-mono text-xs text-muted-foreground">{point.document_family ?? "-"}</td>
              <td className="p-3 font-mono text-xs">{point.version ?? "-"}</td>
              <td className="p-3"><Database className="h-4 w-4 text-muted-foreground" /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
