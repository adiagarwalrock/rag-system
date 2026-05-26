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
      className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)_auto] lg:items-end"
      onSubmit={form.handleSubmit((values) => createClient.mutate(values, { onSuccess: () => form.reset() }))}
    >
      <label className="block text-xs text-muted-foreground">
        Client name
        <input className="control mt-1 w-full" {...form.register("name")} />
      </label>
      <label className="block text-xs text-muted-foreground">
        Description
        <input className="control mt-1 w-full" {...form.register("description")} />
      </label>
      <button className="button-primary mb-0 w-full lg:w-auto" disabled={createClient.isPending}>
        <Plus className="h-4 w-4" />
        Create workspace
      </button>
      {createClient.error && (
        <div className="lg:col-span-3">
          <ErrorState error={createClient.error} title="Create failed" />
        </div>
      )}
    </form>
  );
}
