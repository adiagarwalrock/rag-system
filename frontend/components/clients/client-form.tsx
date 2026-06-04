"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { Plus } from "lucide-react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { useEffect } from "react";
import { useCreateClient } from "@/lib/hooks/use-clients";
import { useEmbeddingModels } from "@/lib/hooks/use-models";
import { DarkSelect } from "@/components/common/dark-select";
import { ErrorState } from "@/components/common/error-state";

const schema = z.object({
  name: z.string().min(2),
  description: z.string().optional(),
  embedding_model: z.string().optional(),
});

type FormValues = z.infer<typeof schema>;

export function ClientForm() {
  const createClient = useCreateClient();
  const { data: embeddingModels = [] } = useEmbeddingModels();

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { name: "", description: "", embedding_model: "" },
  });

  const selectedModel = form.watch("embedding_model") ?? "";

  // API returns models already sorted openai → gemini → other
  const selectOptions = embeddingModels.map((m) => ({
    value: m.id,
    label: m.display_name || m.id,
    hint: `${m.dimensions.toLocaleString()}d`,
  }));

  const defaultModelId = embeddingModels.find((m) => m.default)?.id ?? embeddingModels[0]?.id;

  // Seed the form field once the model list loads (empty string is the pre-load placeholder).
  useEffect(() => {
    if (defaultModelId && !form.getValues("embedding_model")) {
      form.setValue("embedding_model", defaultModelId);
    }
  }, [defaultModelId, form]);

  function handleSubmit(values: FormValues) {
    createClient.mutate(
      { name: values.name, description: values.description, embedding_model: values.embedding_model || undefined },
      { onSuccess: () => form.reset() },
    );
  }

  return (
    <form
      className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)_minmax(0,1fr)_auto] lg:items-end"
      onSubmit={form.handleSubmit(handleSubmit)}
    >
      <label className="block text-xs text-muted-foreground">
        Client name
        <input className="control mt-1 w-full" {...form.register("name")} />
      </label>
      <label className="block text-xs text-muted-foreground">
        Description
        <input className="control mt-1 w-full" {...form.register("description")} />
      </label>
      {selectOptions.length > 0 && (
        <div className="text-xs text-muted-foreground">
          Embedding model
          <DarkSelect
            label="Embedding model"
            className="mt-1"
            value={selectedModel}
            options={selectOptions}
            onChange={(value) => form.setValue("embedding_model", value)}
          />
        </div>
      )}
      <button className="button-primary mb-0 w-full lg:w-auto" disabled={createClient.isPending}>
        <Plus className="h-4 w-4" />
        Create workspace
      </button>
      {createClient.error && (
        <div className="lg:col-span-4">
          <ErrorState error={createClient.error} title="Create failed" />
        </div>
      )}
    </form>
  );
}
