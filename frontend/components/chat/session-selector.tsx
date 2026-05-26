"use client";

import { Plus } from "lucide-react";
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
      <select
        className="control w-52 font-mono text-xs"
        value={sessionId || getStoredSession(workspaceId)}
        onChange={(event) => {
          onChange(event.target.value);
          setStoredSession(workspaceId, event.target.value);
        }}
        disabled={!workspaceId}
        aria-label="Chat session"
      >
        <option value="">No session</option>
        {(sessions.data ?? []).map((session) => (
          <option key={session.id} value={session.id}>
            {session.title || session.id}
          </option>
        ))}
      </select>
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
