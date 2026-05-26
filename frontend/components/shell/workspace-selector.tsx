"use client";

import { DarkSelect } from "@/components/common/dark-select";
import { useClients } from "@/lib/hooks/use-clients";

export function WorkspaceSelector({
  value,
  onChange,
  className = "w-60",
}: {
  value: string;
  onChange: (workspaceId: string) => void;
  className?: string;
}) {
  const clients = useClients();
  return (
    <DarkSelect
      label="Workspace"
      value={value}
      placeholder="Select workspace"
      onChange={onChange}
      className={className}
      buttonClassName="font-mono"
      options={[
        { value: "", label: "Select workspace" },
        ...(clients.data ?? []).map((client) => ({ value: client.id, label: client.name })),
      ]}
    />
  );
}
