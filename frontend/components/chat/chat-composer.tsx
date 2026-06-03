"use client";

import { Check, Plus, Send, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { DarkSelect } from "@/components/common/dark-select";
import type { Client } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";

export function ChatComposer({
  disabled,
  streaming,
  onSubmit,
  onCancel,
  clients,
  activeClientId,
  onClientChange,
  llmModel,
  onLlmModelChange,
  modelOptions,
  reasoningEffort,
  onReasoningEffortChange,
  includeMemory,
  onIncludeMemoryChange,
  includeConflicts,
  onIncludeConflictsChange,
  initialQuestion = "",
}: {
  disabled?: boolean;
  streaming?: boolean;
  onSubmit: (question: string) => void;
  onCancel?: () => void;
  clients?: Client[];
  activeClientId?: string;
  onClientChange?: (clientId: string) => void;
  llmModel?: string;
  onLlmModelChange?: (value: string) => void;
  modelOptions?: Array<{ value: string; label: string }>;
  reasoningEffort?: "low" | "medium" | "high";
  onReasoningEffortChange?: (value: "low" | "medium" | "high") => void;
  includeMemory?: boolean;
  onIncludeMemoryChange?: (value: boolean) => void;
  includeConflicts?: boolean;
  onIncludeConflictsChange?: (value: boolean) => void;
  initialQuestion?: string;
}) {
  const [question, setQuestion] = useState(initialQuestion);
  const [clientMenuOpen, setClientMenuOpen] = useState(false);
  const clientMenuRef = useRef<HTMLDivElement | null>(null);
  const activeClient = clients?.find((client) => client.id === activeClientId);

  useEffect(() => {
    if (!clientMenuOpen) return;

    function handlePointerDown(event: PointerEvent) {
      if (!clientMenuRef.current?.contains(event.target as Node)) {
        setClientMenuOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setClientMenuOpen(false);
    }

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [clientMenuOpen]);

  return (
    <form
      className="rounded-3xl border border-border bg-[#242424] p-3 shadow-2xl shadow-black/35"
      onSubmit={(event) => {
        event.preventDefault();
        if (question.trim()) {
          onSubmit(question.trim());
          setQuestion("");
        }
      }}
    >
      <label className="sr-only" htmlFor="question">Question</label>
      <textarea
        id="question"
        placeholder={activeClient ? `Ask ${activeClient.name}` : "Select a client, then ask across scoped documents"}
        className="max-h-56 min-h-24 w-full resize-none rounded-2xl border-0 bg-transparent px-2 py-2 text-[15px] leading-6 outline-none placeholder:text-muted-foreground"
        value={question}
        disabled={disabled}
        onChange={(event) => setQuestion(event.target.value)}
        onKeyDown={(event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.currentTarget.form?.requestSubmit();
          }
        }}
      />
      <div className="flex items-center justify-between gap-3">
        <div ref={clientMenuRef} className="relative flex items-center gap-2">
          <button
            type="button"
            className={cn(
              "inline-flex h-9 w-9 items-center justify-center rounded-full border border-border bg-background text-muted-foreground transition hover:bg-muted hover:text-foreground",
              clientMenuOpen && "bg-muted text-foreground",
            )}
            onClick={() => setClientMenuOpen((open) => !open)}
            aria-label="Select client workspace"
          >
            <Plus className="h-5 w-5" />
          </button>
          {clientMenuOpen && (
            <div className="absolute bottom-12 left-0 z-40 w-80 rounded-2xl border border-border bg-card p-2 shadow-2xl shadow-black/50">
              <div className="px-3 py-2 text-xs font-medium text-muted-foreground">Client workspace</div>
              <div className="max-h-72 overflow-y-auto">
                {(clients ?? []).length ? (
                  clients?.map((client) => (
                    <button
                      key={client.id}
                      type="button"
                      className="flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-left text-sm hover:bg-muted"
                      onClick={() => {
                        onClientChange?.(client.id);
                        setClientMenuOpen(false);
                      }}
                    >
                      <span className="min-w-0">
                        <span className="block truncate text-foreground">{client.name}</span>
                        <span className="block truncate font-mono text-[11px] text-muted-foreground">{client.id}</span>
                      </span>
                      {client.id === activeClientId && <Check className="h-4 w-4 shrink-0 text-blue-300" />}
                    </button>
                  ))
                ) : (
                  <div className="px-3 py-4 text-sm text-muted-foreground">No clients returned by the API.</div>
                )}
              </div>
              <div className="mt-2 border-t border-border pt-2">
                <div className="px-3 py-2 text-xs font-medium text-muted-foreground">Context options</div>
                <label className="flex cursor-pointer items-center justify-between rounded-xl px-3 py-2 text-sm hover:bg-muted">
                  <span>
                    <span className="block text-foreground">Use memory</span>
                    <span className="block text-xs text-muted-foreground">Include relevant prior session context.</span>
                  </span>
                  <input
                    type="checkbox"
                    checked={includeMemory ?? true}
                    onChange={(event) => onIncludeMemoryChange?.(event.target.checked)}
                  />
                </label>
                <label className="flex cursor-pointer items-center justify-between rounded-xl px-3 py-2 text-sm hover:bg-muted">
                  <span>
                    <span className="block text-foreground">Conflict checks</span>
                    <span className="block text-xs text-muted-foreground">Detect conflicting facts across retrieved sources.</span>
                  </span>
                  <input
                    type="checkbox"
                    checked={includeConflicts ?? true}
                    onChange={(event) => onIncludeConflictsChange?.(event.target.checked)}
                  />
                </label>
              </div>
            </div>
          )}
          <div className="hidden max-w-[220px] items-center gap-1 rounded-full border border-border bg-background px-3 py-2 text-xs text-muted-foreground sm:flex">
            <span className="truncate">{activeClient?.name ?? "No client"}</span>
          </div>
        </div>
        <div className="flex min-w-0 items-center justify-end gap-2">
          {modelOptions && modelOptions.length > 0 && llmModel && (
            <DarkSelect
              label="Model"
              value={llmModel}
              options={modelOptions}
              onChange={(value) => onLlmModelChange?.(value)}
              buttonClassName="min-w-[7.5rem]"
              menuSide="top"
            />
          )}
          <DarkSelect
            label="Reasoning effort"
            value={reasoningEffort ?? "medium"}
            options={[
              { value: "low", label: "Low" },
              { value: "medium", label: "Medium" },
              { value: "high", label: "High" },
            ]}
            onChange={(value) => onReasoningEffortChange?.(value)}
            buttonClassName="min-w-[6rem]"
            menuSide="top"
          />
          {streaming ? (
            <button
              type="button"
              className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-border bg-background text-muted-foreground hover:bg-muted"
              onClick={onCancel}
              aria-label="Cancel response"
            >
              <Square className="h-4 w-4" />
            </button>
          ) : (
            <span className="group relative inline-flex">
              <button
                className="inline-flex h-9 w-9 items-center justify-center rounded-full bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                disabled={disabled || !question.trim()}
                aria-label="Send message"
                aria-describedby="send-message-tooltip"
              >
                <Send className="h-4 w-4" />
              </button>
              <span
                id="send-message-tooltip"
                role="tooltip"
                className="pointer-events-none absolute bottom-11 right-0 hidden whitespace-nowrap rounded-md border border-border bg-card px-2.5 py-1.5 text-xs text-foreground shadow-xl shadow-black/30 group-hover:block group-focus-within:block"
              >
                Send (Cmd/Ctrl + Enter)
              </span>
            </span>
          )}
        </div>
      </div>
    </form>
  );
}
