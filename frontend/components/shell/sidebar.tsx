"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Activity,
  BarChart3,
  ChevronDown,
  Database,
  Edit3,
  FileText,
  History,
  Menu,
  MessageSquare,
  Trash2,
  Users,
} from "lucide-react";
import { useState } from "react";
import { ConfirmDeleteDialog } from "@/components/common/confirm-delete-dialog";
import { useCreateSession, useDeleteSession, useDeleteSessions, useSessions } from "@/lib/hooks/use-chat";
import { useWorkspaceStore } from "@/lib/state/workspace-store";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/history", label: "History", icon: History },
  { href: "/documents", label: "Documents", icon: FileText },
  { href: "/clients", label: "Clients", icon: Users },
];

const moreItems = [
  { href: "/quality", label: "Quality Evaluation", icon: BarChart3, disabled: true },
  { href: "/inspector", label: "Qdrant Inspector", icon: Database },
  { href: "/runtime", label: "Runtime Status", icon: Activity },
];

export function Sidebar({
  width,
  onResizeStart,
  onWidthChange,
}: {
  width: number;
  onResizeStart: (event: React.PointerEvent) => void;
  onWidthChange: (width: number) => void;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false);
  const { workspaceId, sessionId, setSessionId } = useWorkspaceStore();
  const sessions = useSessions(workspaceId);
  const createSession = useCreateSession(workspaceId);
  const deleteSession = useDeleteSession(workspaceId);
  const deleteSessions = useDeleteSessions(workspaceId);
  const moreActive = moreItems.some((item) => pathname === item.href);
  const chatActive = pathname === "/" || pathname.startsWith("/c/");

  function afterSessionDelete(deletedSessionIds: string[]) {
    if (deletedSessionIds.includes(sessionId)) {
      setSessionId("");
      router.push("/");
    }
    setOpen(false);
  }

  const content = (
    <aside
      className="relative flex h-full flex-col border-r border-border bg-card/80 px-3 py-4 backdrop-blur"
      style={{ width: `min(${width}px, 85vw)` }}
    >
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
                router.push(`/c/${encodeURIComponent(session.id)}`);
                setOpen(false);
              },
            });
          }}
        >
          <Edit3 className="h-4 w-4" />
          New chat
        </button>
        {navItems.map((item) => {
          const active = item.href === "/" ? chatActive : pathname === item.href;
          const Icon = item.icon;
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
        <div>
          <button
            type="button"
            className={cn(
              "flex h-9 w-full items-center gap-2 rounded-md px-2 text-sm text-muted-foreground transition hover:bg-muted hover:text-foreground",
              moreActive && "bg-muted text-foreground",
            )}
            aria-expanded={moreOpen}
            onClick={() => setMoreOpen((current) => !current)}
          >
            <Menu className="h-4 w-4" />
            <span className="min-w-0 flex-1 text-left">More</span>
            <ChevronDown className={cn("h-3.5 w-3.5 transition", moreOpen && "rotate-180")} />
          </button>
          {moreOpen ? (
            <div className="mt-1 space-y-1 pl-3">
              {moreItems.map((item) => {
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
            </div>
          ) : null}
        </div>
      </nav>

      <div className="relative mt-6 min-h-0 flex-1">
        <div className="mb-2 flex items-center justify-between gap-2 px-2">
          <div className="text-xs font-semibold text-muted-foreground">Sessions</div>
          {sessions.data?.length ? (
            <ConfirmDeleteDialog
              title="Delete all sessions"
              description="This deletes every chat session in the active workspace, including message history."
              pending={deleteSessions.isPending}
              onConfirm={() => {
                const ids = (sessions.data ?? []).map((session) => session.id);
                deleteSessions.mutate(ids, {
                  onSuccess: () => afterSessionDelete(ids),
                });
              }}
            >
              <button
                className="button-ghost h-7 px-2 text-[11px] text-red-300 hover:text-red-200"
                disabled={deleteSessions.isPending}
              >
                Delete all
              </button>
            </ConfirmDeleteDialog>
          ) : null}
        </div>
        <div className="no-scrollbar h-[calc(100%-1.5rem)] space-y-1 overflow-y-auto pb-10 pr-1">
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
                <div
                  key={session.id}
                  className={cn(
                    "group flex items-center gap-1 rounded-md text-muted-foreground transition hover:bg-muted hover:text-foreground",
                    active && "bg-muted text-foreground",
                  )}
                >
                  <Link
                    href={`/c/${encodeURIComponent(session.id)}`}
                    className="min-w-0 flex-1 px-2 py-2 text-left text-sm"
                    onClick={() => {
                      setSessionId(session.id);
                      setOpen(false);
                    }}
                  >
                    <span className="block truncate">
                      {session.title || "Untitled session"}
                    </span>
                  </Link>
                  <ConfirmDeleteDialog
                    title="Delete session"
                    description="This deletes the selected chat session and its message history."
                    pending={deleteSession.isPending}
                    onConfirm={() => {
                      deleteSession.mutate(session.id, {
                        onSuccess: () => afterSessionDelete([session.id]),
                      });
                    }}
                  >
                    <button
                      className="mr-1 hidden h-7 w-7 items-center justify-center rounded-md text-red-300 hover:bg-destructive/10 hover:text-red-200 group-hover:inline-flex focus:inline-flex"
                      disabled={deleteSession.isPending}
                      aria-label="Delete session"
                      onClick={(event) => event.stopPropagation()}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </ConfirmDeleteDialog>
                </div>
              );
            })
          ) : (
            <div className="rounded-md px-2 py-2 text-xs text-muted-foreground">
              No sessions yet.
            </div>
          )}
        </div>
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-card via-card/80 to-transparent" />
      </div>
      <div
        role="separator"
        aria-label="Resize sidebar"
        aria-orientation="vertical"
        tabIndex={0}
        className="absolute inset-y-0 -right-1 hidden w-2 cursor-col-resize touch-none items-center justify-center md:flex"
        onPointerDown={onResizeStart}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") {
            event.preventDefault();
            onWidthChange(width - 16);
          }
          if (event.key === "ArrowRight") {
            event.preventDefault();
            onWidthChange(width + 16);
          }
        }}
      >
        <span className="h-10 w-px rounded-full bg-border transition hover:bg-primary" />
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
