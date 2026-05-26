"use client";

import { ClientCard } from "@/components/clients/client-card";
import { ClientForm } from "@/components/clients/client-form";
import { EmptyState } from "@/components/common/empty-state";
import { ErrorState } from "@/components/common/error-state";
import { LoadingState } from "@/components/common/loading-state";
import { SectionCard } from "@/components/common/section-card";
import { PageHeader } from "@/components/shell/page-header";
import { useClients, useDeleteClient } from "@/lib/hooks/use-clients";

export default function ClientsPage() {
  const clients = useClients();
  const deleteClient = useDeleteClient();

  return (
    <div>
      <PageHeader
        title="Workspaces."
        description="Create isolated client workspaces with scoped documents, sessions, history, and semantic memory."
      />
      <div className="mx-auto max-w-5xl space-y-4">
        <SectionCard
          title="Create workspace"
          description="Add a new isolated client scope."
        >
          <ClientForm />
        </SectionCard>

        <div>
          {clients.isLoading ? <LoadingState /> : clients.error ? <ErrorState error={clients.error} /> : clients.data?.length ? (
            <SectionCard title={`Workspace library // ${clients.data.length}`}>
              <div className="space-y-3">
                {clients.data.map((client) => (
                  <ClientCard
                    key={client.id}
                    client={client}
                    pending={deleteClient.isPending}
                    onDelete={() => deleteClient.mutate(client.id)}
                  />
                ))}
              </div>
            </SectionCard>
          ) : <EmptyState title="No workspaces" description="Create a workspace before uploading documents or asking scoped questions." />}
        </div>
      </div>
    </div>
  );
}
