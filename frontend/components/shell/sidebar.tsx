"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Activity,
  BarChart3,
  Database,
  Edit3,
  FileText,
  History,
  Menu,
  MessageSquare,
  Users,
} from "lucide-react";
import { useState } from "react";
import { useCreateSession, useSessions } from "@/lib/hooks/use-chat";
import { useWorkspaceStore } from "@/lib/state/workspace-store";
import { cn, truncate } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/history", label: "History", icon: History },
  { href: "/documents", label: "Documents", icon: FileText },
  { href: "/clients", label: "Clients", icon: Users },
  { href: "/quality", label: "Quality Evaluation", icon: BarChart3, disabled: true },
  { href: "/inspector", label: "Qdrant Inspector", icon: Database },
  { href: "/runtime", label: "Runtime Status", icon: Activity },
];

export function Sidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const { workspaceId, sessionId, setSessionId } = useWorkspaceStore();
  const sessions = useSessions(workspaceId);
  const createSession = useCreateSession(workspaceId);

  const content = (
    <aside className="flex h-full w-72 flex-col border-r border-border bg-card/80 px-3 py-4 backdrop-blur">
      <div className="px-2">
        <div className="text-sm font-semibold">RAG Console</div>
        <div className="font-mono text-[11px] text-muted-foreground">internal ops</div>
      </div>

      <nav className="mt-5 space-y-1">
        <button
          className="mb-2 flex h-9 w-full items-center gap-2 rounded-md bg-muted px-2 text-sm font-medium text-foreground transition hover:bg-muted/80 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={!workspaceId || createSession.isPending}
          onClick={() => {
            createSession.mutate(undefined, {
              onSuccess: (session) => {
                setSessionId(session.id);
                router.push("/");
                setOpen(false);
              },
            });
          }}
        >
          <Edit3 className="h-4 w-4" />
          New chat
        </button>
        {navItems.map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          if (item.disabled) {
            return (
              <div
                key={item.href}
                aria-disabled="true"
                className="flex h-9 cursor-not-allowed items-center gap-2 rounded-md px-2 text-sm text-muted-foreground/55"
              >
                <Icon className="h-4 w-4" />
                <span className="min-w-0 flex-1 truncate">{item.label}</span>
                <span className="rounded-md border border-warning/30 bg-warning/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-300">
                  ToDo
                </span>
              </div>
            );
          }
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setOpen(false)}
              className={cn(
                "flex h-9 items-center gap-2 rounded-md px-2 text-sm text-muted-foreground transition hover:bg-muted hover:text-foreground",
                active && "bg-muted text-foreground",
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-6 min-h-0 flex-1 overflow-y-auto pr-1">
        <div className="mb-2 px-2 text-xs font-semibold text-muted-foreground">
          Sessions
        </div>
        <div className="space-y-1">
          {!workspaceId ? (
            <div className="rounded-md px-2 py-2 text-xs text-muted-foreground">
              Select a client from the composer.
            </div>
          ) : sessions.isLoading ? (
            <div className="px-2 py-2 text-xs text-muted-foreground">Loading sessions...</div>
          ) : sessions.data?.length ? (
            sessions.data.map((session) => {
              const active = session.id === sessionId;
              return (
                <button
                  key={session.id}
                  className={cn(
                    "block w-full rounded-md px-2 py-2 text-left text-sm text-muted-foreground transition hover:bg-muted hover:text-foreground",
                    active && "bg-muted text-foreground",
                  )}
                  onClick={() => {
                    setSessionId(session.id);
                    router.push("/");
                    setOpen(false);
                  }}
                >
                  <span className="block truncate">
                    {truncate(session.title || "Untitled session", 34)}
                  </span>
                </button>
              );
            })
          ) : (
            <div className="rounded-md px-2 py-2 text-xs text-muted-foreground">
              No sessions yet.
            </div>
          )}
        </div>
      </div>
    </aside>
  );

  return (
    <>
      <button
        className="fixed left-3 top-3 z-40 inline-flex h-9 w-9 items-center justify-center rounded-md border border-border bg-card md:hidden"
        onClick={() => setOpen(true)}
        aria-label="Open navigation"
      >
        <Menu className="h-4 w-4" />
      </button>
      <div className="fixed inset-y-0 left-0 z-30 hidden md:block">{content}</div>
      {open && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div className="absolute inset-0 bg-black/70" onClick={() => setOpen(false)} />
          <div className="absolute inset-y-0 left-0">{content}</div>
        </div>
      )}
    </>
  );
}
