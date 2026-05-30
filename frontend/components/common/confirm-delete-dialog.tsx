"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { Trash2, X } from "lucide-react";
import { useState } from "react";

export function ConfirmDeleteDialog({
  title,
  description,
  onConfirm,
  children,
  pending,
}: {
  title: string;
  description: string;
  onConfirm: () => void | Promise<void>;
  children: React.ReactNode;
  pending?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const busy = Boolean(pending || confirming);

  async function handleConfirm() {
    if (busy) return;
    try {
      const result = onConfirm();
      if (result instanceof Promise) {
        setConfirming(true);
        await result;
      }
      setOpen(false);
    } finally {
      setConfirming(false);
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger asChild>{children}</Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/70" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[92vw] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-5 shadow-xl">
          <div className="flex items-start justify-between gap-4">
            <div>
              <Dialog.Title className="text-base font-semibold">{title}</Dialog.Title>
              <Dialog.Description className="mt-2 text-sm text-muted-foreground">
                {description}
              </Dialog.Description>
            </div>
            <Dialog.Close className="button-ghost h-8 w-8 p-0" aria-label="Close">
              <X className="h-4 w-4" />
            </Dialog.Close>
          </div>
          <div className="mt-5 flex justify-end gap-2">
            <Dialog.Close className="button-secondary" disabled={busy}>Cancel</Dialog.Close>
            <button
              className="button-primary bg-destructive hover:bg-destructive/90"
              onClick={handleConfirm}
              disabled={busy}
            >
              <Trash2 className="h-4 w-4" />
              {busy ? "Deleting..." : "Delete"}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
