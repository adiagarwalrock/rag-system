"use client";

import { useClients } from "@/lib/hooks/use-clients";

export function WorkspaceSelector({
  value,
  onChange,
  className = "control w-60 font-mono text-xs",
}: {
  value: string;
  onChange: (workspaceId: string) => void;
  className?: string;
}) {
  const clients = useClients();
  return (
    <select className={className} value={value} onChange={(event) => onChange(event.target.value)} aria-label="Workspace">
      <option value="">Select workspace</option>
      {(clients.data ?? []).map((client) => (
        <option key={client.id} value={client.id}>{client.name}</option>
      ))}
    </select>
  );
}
