"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { Plus } from "lucide-react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { useCreateClient } from "@/lib/hooks/use-clients";
import { ErrorState } from "@/components/common/error-state";

const schema = z.object({
  name: z.string().min(2),
  description: z.string().optional(),
});

export function ClientForm() {
  const createClient = useCreateClient();
  const form = useForm<z.infer<typeof schema>>({
    resolver: zodResolver(schema),
    defaultValues: { name: "", description: "" },
  });

  return (
    <form
      className="space-y-4"
      onSubmit={form.handleSubmit((values) => createClient.mutate(values, { onSuccess: () => form.reset() }))}
    >
      <label className="block text-xs text-muted-foreground">
        Client name
        <input className="control mt-1 w-full" {...form.register("name")} />
      </label>
      <label className="block text-xs text-muted-foreground">
        Description
        <textarea className="mt-1 min-h-28 w-full resize-y rounded-md border border-border bg-muted/70 p-3 text-sm text-foreground outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20" {...form.register("description")} />
      </label>
      <button className="button-primary w-full" disabled={createClient.isPending}>
        <Plus className="h-4 w-4" />
        Create workspace
      </button>
      {createClient.error && <ErrorState error={createClient.error} title="Create failed" />}
    </form>
  );
}
