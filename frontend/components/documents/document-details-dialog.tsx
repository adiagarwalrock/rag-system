"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import type { Document } from "@/lib/api/schemas";
import { JsonViewer } from "@/components/common/json-viewer";
import { StatusBadge } from "@/components/common/status-badge";
import { formatNumber } from "@/lib/utils";

export function DocumentDetailsDialog({
  document,
  children,
}: {
  document: Document;
  children: React.ReactNode;
}) {
  return (
    <Dialog.Root>
      <Dialog.Trigger asChild>{children}</Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/70" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[86vh] w-[94vw] max-w-3xl -translate-x-1/2 -translate-y-1/2 overflow-auto rounded-lg border border-border bg-card p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <Dialog.Title className="text-base font-semibold">{document.filename}</Dialog.Title>
              <Dialog.Description className="mt-1 font-mono text-xs text-muted-foreground">{document.id}</Dialog.Description>
            </div>
            <Dialog.Close className="button-ghost h-8 w-8 p-0" aria-label="Close">
              <X className="h-4 w-4" />
            </Dialog.Close>
          </div>
          <div className="mt-5 grid gap-3 md:grid-cols-3">
            <Fact label="Status" value={<StatusBadge status={document.status} />} />
            <Fact label="Document family" value={document.document_family} mono />
            <Fact label="Version" value={document.version} mono />
            <Fact label="Parser" value={document.parser_used ?? "-"} mono />
            <Fact label="Vector points" value={formatNumber(document.vector_points)} mono />
            <Fact label="Chunks" value={formatNumber(document.chunk_count)} mono />
            <Fact label="Collection" value={document.active_collection ?? "-"} mono />
          </div>
          {document.last_error && <p className="mt-4 rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-red-200">{document.last_error}</p>}
          <div className="mt-4">
            <h3 className="mb-2 text-sm font-medium">Raw metadata</h3>
            <JsonViewer value={document.metadata ?? document} />
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function Fact({ label, value, mono }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="rounded-md border border-border bg-muted/30 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={mono ? "mt-1 font-mono text-xs" : "mt-1 text-sm"}>{value}</div>
    </div>
  );
}
