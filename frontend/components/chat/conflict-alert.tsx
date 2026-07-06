import { AlertTriangle } from "lucide-react";
import type { Conflict } from "@/lib/api/schemas";
import { StatusBadge } from "@/components/common/status-badge";

export function ConflictAlert({ conflict }: { conflict: Conflict }) {
  return (
    <div className="rounded-lg border border-warning/30 bg-warning/10 p-3">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 text-amber-300" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap gap-2">
            <StatusBadge status="warning" label={conflict.severity} />
            <span className="rounded-md border border-border bg-background px-2 py-1 font-mono text-xs text-muted-foreground">
              {conflict.type}
            </span>
          </div>
          <p className="mt-2 text-sm text-foreground">{conflict.explanation}</p>
          {conflict.values?.length ? (
            <p className="mt-2 font-mono text-xs text-muted-foreground">{conflict.values.join(" vs ")}</p>
          ) : null}
          {conflict.documents?.length ? (
            <p className="mt-1 text-xs text-muted-foreground">Documents: {conflict.documents.join(", ")}</p>
          ) : null}
        </div>
      </div>
    </div>
  );
}
