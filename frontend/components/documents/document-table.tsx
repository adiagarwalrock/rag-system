"use client";

import { Eye, RotateCcw, Trash2 } from "lucide-react";
import type { Document } from "@/lib/api/schemas";
import { ConfirmDeleteDialog } from "@/components/common/confirm-delete-dialog";
import { StatusBadge } from "@/components/common/status-badge";
import { DocumentDetailsDialog } from "@/components/documents/document-details-dialog";
import { formatDate, formatNumber } from "@/lib/utils";

export function DocumentTable({
  documents,
  onDelete,
  onRetry,
  pending,
}: {
  documents: Document[];
  onDelete: (documentId: string) => void;
  onRetry: (documentId: string) => void;
  pending?: boolean;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[860px] text-left text-sm">
        <thead className="border-b border-border text-xs text-muted-foreground">
          <tr>
            <th className="py-2 pr-3 font-medium">Filename</th>
            <th className="py-2 pr-3 font-medium">Status</th>
            <th className="py-2 pr-3 font-medium">Version</th>
            <th className="py-2 pr-3 font-medium">Family</th>
            <th className="py-2 pr-3 font-medium">Parser</th>
            <th className="py-2 pr-3 font-medium">Vectors</th>
            <th className="py-2 pr-3 font-medium">Uploaded</th>
            <th className="py-2 text-right font-medium">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {documents.map((document) => (
            <tr key={document.id} className="align-middle">
              <td className="max-w-xs truncate py-3 pr-3 font-medium">{document.filename}</td>
              <td className="py-3 pr-3"><StatusBadge status={document.status} /></td>
              <td className="py-3 pr-3 font-mono text-xs">{document.version}</td>
              <td className="max-w-[180px] truncate py-3 pr-3 font-mono text-xs text-muted-foreground">{document.document_family}</td>
              <td className="py-3 pr-3 font-mono text-xs text-muted-foreground">{document.parser_used ?? "-"}</td>
              <td className="py-3 pr-3 font-mono text-xs">{formatNumber(document.vector_points)}</td>
              <td className="py-3 pr-3 font-mono text-xs text-muted-foreground">{formatDate(document.uploaded_at)}</td>
              <td className="py-3">
                <div className="flex justify-end gap-1">
                  <DocumentDetailsDialog document={document}>
                    <button className="button-ghost h-8 w-8 p-0" aria-label="Details"><Eye className="h-4 w-4" /></button>
                  </DocumentDetailsDialog>
                  <button className="button-ghost h-8 w-8 p-0" aria-label="Retry" onClick={() => onRetry(document.id)} disabled={pending}><RotateCcw className="h-4 w-4" /></button>
                  <ConfirmDeleteDialog title="Delete document" description="This removes the document and associated vector references." onConfirm={() => onDelete(document.id)} pending={pending}>
                    <button className="button-ghost h-8 w-8 p-0 text-red-300" aria-label="Delete"><Trash2 className="h-4 w-4" /></button>
                  </ConfirmDeleteDialog>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
