"use client";

import { useEffect } from "react";
import { EmptyState } from "@/components/common/empty-state";
import { ErrorState } from "@/components/common/error-state";
import { LoadingState } from "@/components/common/loading-state";
import { DarkSelect } from "@/components/common/dark-select";
import { QueryHistoryTable } from "@/components/history/query-history-table";
import { PageHeader } from "@/components/shell/page-header";
import { useClients } from "@/lib/hooks/use-clients";
import { useQueryHistory } from "@/lib/hooks/use-query-history";
import { useWorkspaceStore } from "@/lib/state/workspace-store";

export default function QueryHistoryPage() {
  const clients = useClients();
  const { workspaceId, setWorkspaceId } = useWorkspaceStore();
  const history = useQueryHistory(workspaceId);

  useEffect(() => {
    if (!workspaceId && clients.data?.[0]) setWorkspaceId(clients.data[0].id);
  }, [clients.data, setWorkspaceId, workspaceId]);

  return (
    <div>
      <PageHeader
        title="History."
        description="Review previous questions, answers, evidence, retrieval scores, latency, and conflict flags."
        actions={
          <DarkSelect
            label="Workspace"
            value={workspaceId}
            placeholder="Select workspace"
            onChange={setWorkspaceId}
            className="w-60"
            buttonClassName="font-mono"
            options={[
              { value: "", label: "Select workspace" },
              ...(clients.data ?? []).map((client) => ({ value: client.id, label: client.name })),
            ]}
          />
        }
      />
      {history.isLoading ? <LoadingState /> : history.error ? <ErrorState error={history.error} /> : history.data?.length ? <QueryHistoryTable rows={history.data} /> : <EmptyState title="No history" description="Completed RAG queries for this workspace will appear here." />}
    </div>
  );
}
