"use client";

import { Plus } from "lucide-react";
import { DarkSelect } from "@/components/common/dark-select";
import { useCreateSession, useSessions } from "@/lib/hooks/use-chat";
import { getStoredSession, setStoredSession } from "@/lib/state/workspace-store";

export function SessionSelector({
  workspaceId,
  sessionId,
  onChange,
}: {
  workspaceId: string;
  sessionId: string;
  onChange: (sessionId: string) => void;
}) {
  const sessions = useSessions(workspaceId);
  const createSession = useCreateSession(workspaceId);

  return (
    <div className="flex items-center gap-2">
      <DarkSelect
        label="Chat session"
        value={sessionId || getStoredSession(workspaceId)}
        onChange={(value) => {
          onChange(value);
          setStoredSession(workspaceId, value);
        }}
        disabled={!workspaceId}
        className="w-52"
        buttonClassName="font-mono"
        options={[
          { value: "", label: "No session" },
          ...(sessions.data ?? []).map((session) => ({
            value: session.id,
            label: session.title || session.id,
          })),
        ]}
      />
      <button
        className="button-secondary"
        disabled={!workspaceId || createSession.isPending}
        onClick={() => createSession.mutate(undefined, {
          onSuccess: (session) => {
            onChange(session.id);
            setStoredSession(workspaceId, session.id);
          },
        })}
      >
        <Plus className="h-4 w-4" />
        New
      </button>
    </div>
  );
}
