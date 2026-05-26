import { RotateCcw } from "lucide-react";
import type { IngestionJob } from "@/lib/api/schemas";
import { StatusBadge } from "@/components/common/status-badge";
import { ParserFallbackChain } from "@/components/documents/parser-fallback-chain";
import { formatDate, formatNumber } from "@/lib/utils";

export function IngestionJobCard({
  job,
  onRetry,
  retrying,
}: {
  job: IngestionJob;
  onRetry?: () => void;
  retrying?: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-medium">{job.filename}</h3>
          <div className="mt-2 flex flex-wrap gap-2">
            <StatusBadge status={job.status} />
            {job.parser_attempted && <span className="rounded-md border border-border bg-muted px-2 py-1 font-mono text-xs text-muted-foreground">{job.parser_attempted}</span>}
            {job.fallback_stage && <span className="rounded-md border border-warning/30 bg-warning/10 px-2 py-1 text-xs text-amber-300">{job.fallback_stage}</span>}
          </div>
        </div>
        <button className="button-secondary" onClick={onRetry} disabled={!onRetry || retrying}>
          <RotateCcw className="h-4 w-4" />
          Retry
        </button>
      </div>
      <div className="mt-4 grid gap-3 text-xs md:grid-cols-4">
        <div><span className="text-muted-foreground">Phase</span><div className="mt-1 font-mono">{job.current_phase ?? "-"}</div></div>
        <div><span className="text-muted-foreground">Version</span><div className="mt-1 font-mono">{job.version ?? "-"}</div></div>
        <div><span className="text-muted-foreground">Vector points</span><div className="mt-1 font-mono">{formatNumber(job.vector_points_created)}</div></div>
        <div><span className="text-muted-foreground">Updated</span><div className="mt-1 font-mono">{formatDate(job.updated_at ?? job.created_at)}</div></div>
      </div>
      {job.error && <p className="mt-3 rounded-md border border-destructive/30 bg-destructive/10 p-2 text-xs text-red-200">{job.error}</p>}
      <div className="mt-4"><ParserFallbackChain /></div>
    </div>
  );
}
