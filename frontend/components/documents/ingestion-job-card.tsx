"use client";

import { RotateCcw } from "lucide-react";
import { useState } from "react";
import type { IngestionJob } from "@/lib/api/schemas";
import { DarkSelect } from "@/components/common/dark-select";
import { StatusBadge } from "@/components/common/status-badge";
import { useParsers } from "@/lib/hooks/use-documents";
import { resolveEmbedLabel, useEmbeddingModels } from "@/lib/hooks/use-models";
import { formatDate, formatNumber } from "@/lib/utils";

const DEFAULT_PARSER_OPTIONS = [
  { value: "auto", label: "Auto (recommended)" },
  { value: "legacy", label: "Legacy" },
];

export function IngestionJobCard({
  job,
  onRetry,
  retrying,
}: {
  job: IngestionJob;
  onRetry?: (parser?: string) => void;
  retrying?: boolean;
}) {
  const [selectedParser, setSelectedParser] = useState<string>(job.parser_attempted ?? "auto");
  const parsers = useParsers();
  const { data: embeddingModels = [] } = useEmbeddingModels();

  const parserOptions = parsers.data?.parsers
    ? parsers.data.parsers.map((p) => ({
        value: p.id,
        label: p.label,
        disabled: !p.available,
        hint: p.available ? undefined : "not configured",
      }))
    : DEFAULT_PARSER_OPTIONS;

  const embedLabel = resolveEmbedLabel(job.embedding_model ?? null, embeddingModels);

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-medium">{job.filename}</h3>
          <div className="mt-2 flex flex-wrap gap-2">
            <StatusBadge status={job.status} />
            {job.parser_attempted && (
              <span className="rounded-md border border-border bg-muted px-2 py-1 font-mono text-xs text-muted-foreground">
                {job.parser_attempted}
              </span>
            )}
          </div>
        </div>
        {job.status !== "failed" && (
          <button className="button-secondary" onClick={() => onRetry?.()} disabled={!onRetry || retrying}>
            <RotateCcw className="h-4 w-4" />
            Retry
          </button>
        )}
      </div>

      <div className="mt-4 grid gap-3 text-xs md:grid-cols-4">
        <div>
          <span className="text-muted-foreground">Phase</span>
          <div className="mt-1 font-mono">{job.current_phase ?? "-"}</div>
        </div>
        <div>
          <span className="text-muted-foreground">Version</span>
          <div className="mt-1 font-mono">{job.version ?? "-"}</div>
        </div>
        <div>
          <span className="text-muted-foreground">Vector points</span>
          <div className="mt-1 font-mono">{formatNumber(job.vector_points_created)}</div>
        </div>
        <div>
          <span className="text-muted-foreground">Embed</span>
          <div className="mt-1 font-mono text-muted-foreground">{embedLabel ?? "-"}</div>
        </div>
      </div>

      {job.error && (
        <p className="mt-3 rounded-md border border-destructive/30 bg-destructive/10 p-2 text-xs text-red-200">
          {job.error}
        </p>
      )}

      {job.status === "failed" && (
        <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-border pt-4">
          <DarkSelect
            label="Retry with parser"
            value={selectedParser}
            options={parserOptions}
            onChange={setSelectedParser}
            className="w-52"
            menuSide="top"
          />
          <button
            className="button-secondary"
            onClick={() => onRetry?.(selectedParser)}
            disabled={!onRetry || retrying}
          >
            <RotateCcw className="h-4 w-4" />
            Retry
          </button>
        </div>
      )}
    </div>
  );
}
