"use client";

import { RefreshCw } from "lucide-react";
import { useState } from "react";
import { EmptyState } from "@/components/common/empty-state";
import { ErrorState } from "@/components/common/error-state";
import { LoadingState } from "@/components/common/loading-state";
import { RuntimeDiagnostics } from "@/components/runtime/runtime-diagnostics";
import { PageHeader } from "@/components/shell/page-header";
import { useRuntimeStatus } from "@/lib/hooks/use-runtime-status";

export default function RuntimePage() {
  const [autoRefresh, setAutoRefresh] = useState(false);
  const runtime = useRuntimeStatus(autoRefresh);

  return (
    <div>
      <PageHeader
        title="Runtime status."
        description="Parser health, database connectivity, Qdrant collections, and AI provider availability."
        actions={
          <>
            <label className="flex h-9 items-center gap-2 rounded-md border border-border bg-muted px-3 text-sm">
              <input type="checkbox" checked={autoRefresh} onChange={(event) => setAutoRefresh(event.target.checked)} />
              Auto-refresh
            </label>
            <button className="button-secondary" onClick={() => runtime.refetch()}>
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
          </>
        }
      />
      {runtime.isLoading ? <LoadingState /> : runtime.error ? <ErrorState error={runtime.error} /> : runtime.data ? <RuntimeDiagnostics status={runtime.data} /> : <EmptyState title="Runtime status unavailable" />}
    </div>
  );
}
