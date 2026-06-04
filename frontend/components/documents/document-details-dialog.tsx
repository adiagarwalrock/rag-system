"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { Copy, X } from "lucide-react";
import { useState } from "react";
import type { Document } from "@/lib/api/schemas";
import { JsonViewer } from "@/components/common/json-viewer";
import { StatusBadge } from "@/components/common/status-badge";
import { useClients } from "@/lib/hooks/use-clients";
import { resolveEmbedLabel, useEmbeddingModels } from "@/lib/hooks/use-models";
import { cn, formatNumber } from "@/lib/utils";

export function DocumentDetailsDialog({
  document,
  children,
}: {
  document: Document;
  children: React.ReactNode;
}) {
  const [copied, setCopied] = useState(false);
  const [metaOpen, setMetaOpen] = useState(false);
  const clients = useClients();
  const { data: embeddingModels = [] } = useEmbeddingModels();

  const clientName = clients.data?.find((c) => c.id === document.client_id)?.name ?? document.client_id;
  const embedLabel = resolveEmbedLabel(document.embedding_model ?? null, embeddingModels) ?? document.embedding_model ?? "-";

  function handleCopy() {
    const text = JSON.stringify(document.metadata ?? document, null, 2);
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

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
            <Fact label="Document family" value={document.document_family} mono colSpan />
            <Fact label="Version" value={document.version} mono />
            <Fact label="Parser" value={document.parser_used ?? "-"} mono />
            <Fact label="Vector points" value={formatNumber(document.vector_points)} mono />
            <Fact label="Embedding model" value={embedLabel} mono />
            <Fact label="Client" value={clientName} mono />
          </div>

          {document.last_error && (
            <p className="mt-4 rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-red-200">
              {document.last_error}
            </p>
          )}

          <div className="mt-4 rounded-md border border-border">
            <button
              type="button"
              onClick={() => setMetaOpen((o) => !o)}
              className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium"
            >
              <span>Raw metadata</span>
              <button
                type="button"
                className="button-ghost flex h-7 items-center gap-1.5 px-2 text-xs"
                onClick={(e) => { e.stopPropagation(); handleCopy(); }}
              >
                <Copy className="h-3.5 w-3.5" />
                {copied ? "Copied!" : "Copy"}
              </button>
            </button>
            <div
              className={cn(
                "overflow-hidden transition-all duration-300 ease-in-out",
                metaOpen ? "max-h-[600px] opacity-100" : "max-h-0 opacity-0",
              )}
            >
              <div className="border-t border-border p-4">
                <JsonViewer value={document.metadata ?? document} />
              </div>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function Fact({
  label,
  value,
  mono,
  colSpan,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
  colSpan?: boolean;
}) {
  return (
    <div className={cn("rounded-md border border-border bg-muted/30 p-3", colSpan && "md:col-span-2")}>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={mono ? "mt-1 break-all font-mono text-xs" : "mt-1 text-sm"}>{value ?? "-"}</div>
    </div>
  );
}
