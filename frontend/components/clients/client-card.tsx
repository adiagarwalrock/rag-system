import { Trash2 } from "lucide-react";
import type { Client } from "@/lib/api/schemas";
import { ConfirmDeleteDialog } from "@/components/common/confirm-delete-dialog";
import { formatDate, formatNumber } from "@/lib/utils";

export function ClientCard({
  client,
  onDelete,
  pending,
}: {
  client: Client;
  onDelete: () => void;
  pending?: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(420px,1fr)_auto] xl:items-start">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold">{client.name}</h3>
          <p className="mt-1 text-sm text-muted-foreground">{client.description || "No description"}</p>
          <p className="mt-2 break-all font-mono text-xs text-muted-foreground">{client.id}</p>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs md:grid-cols-4">
          <Metric label="Documents" value={formatNumber(client.document_count)} />
          <Metric label="Queries" value={formatNumber(client.query_count)} />
          <Metric label="Sessions" value={formatNumber(client.session_count)} />
          <Metric label="Memory" value={formatNumber(client.memory_point_count)} />
        </div>
        <ConfirmDeleteDialog
          title="Delete workspace"
          description="Deleting a workspace cascades documents, history, chat memory, and vector metadata."
          onConfirm={onDelete}
          pending={pending}
        >
          <button className="button-ghost h-8 w-8 p-0 text-red-300" aria-label="Delete workspace">
            <Trash2 className="h-4 w-4" />
          </button>
        </ConfirmDeleteDialog>
      </div>
      <div className="mt-3 font-mono text-xs text-muted-foreground">created {formatDate(client.created_at)}</div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-muted/30 p-2">
      <div className="text-muted-foreground">{label}</div>
      <div className="mt-1 font-mono text-foreground">{value}</div>
    </div>
  );
}
