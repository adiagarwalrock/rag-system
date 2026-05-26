"use client";

import { Check, Copy, X } from "lucide-react";
import { useState } from "react";
import { JsonViewer } from "@/components/common/json-viewer";

export function JsonDrawer({
  open,
  title,
  value,
  onClose,
}: {
  open: boolean;
  title: string;
  value: unknown;
  onClose: () => void;
}) {
  const [wrap, setWrap] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!open) return null;
  return (
    <div className="fixed inset-y-0 right-0 z-50 flex h-screen w-full max-w-3xl flex-col border-l border-border bg-card shadow-2xl">
      <div className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-border px-4">
        <h2 className="min-w-0 truncate font-mono text-sm font-semibold">{title}</h2>
        <div className="flex items-center gap-2">
          <label className="flex h-8 items-center gap-2 rounded-md border border-border bg-muted px-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={wrap}
              onChange={(event) => setWrap(event.target.checked)}
            />
            Wrap
          </label>
          <button
            className="button-ghost h-8 w-8 p-0"
            onClick={async () => {
              await navigator.clipboard.writeText(JSON.stringify(value, null, 2));
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1200);
            }}
            aria-label="Copy JSON"
          >
            {copied ? <Check className="h-4 w-4 text-green-300" /> : <Copy className="h-4 w-4" />}
          </button>
          <button className="button-ghost h-8 w-8 p-0" onClick={onClose} aria-label="Close drawer">
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-hidden p-4">
        <JsonViewer value={value} wrap={wrap} fullHeight />
      </div>
    </div>
  );
}
