"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { Plus } from "lucide-react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { useEffect } from "react";
import { useCreateClient } from "@/lib/hooks/use-clients";
import { useEmbeddingModels, useLLMModels } from "@/lib/hooks/use-models";
import { DarkSelect } from "@/components/common/dark-select";
import { ErrorState } from "@/components/common/error-state";

const schema = z.object({
  name: z.string().min(2),
  description: z.string().optional(),
  embedding_model: z.string().optional(),
  llm_model: z.string().optional(),
});

type FormValues = z.infer<typeof schema>;

export function ClientForm() {
  const createClient = useCreateClient();
  const { data: embeddingModels = [] } = useEmbeddingModels();
  const { data: llmModels = [] } = useLLMModels();

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { name: "", description: "", embedding_model: "", llm_model: "" },
  });

  const selectedEmbedModel = form.watch("embedding_model") ?? "";
  const selectedLLMModel = form.watch("llm_model") ?? "";

  const embedOptions = embeddingModels.map((m) => ({
    value: m.id,
    label: m.display_name || m.id,
    hint: `${m.dimensions.toLocaleString()}d`,
  }));

  const llmOptions = llmModels.map((m) => ({
    value: m.id,
    label: m.display_name || m.id,
    hint: m.supports_reasoning ? "reasoning" : undefined,
  }));

  const defaultEmbedId = embeddingModels.find((m) => m.default)?.id ?? embeddingModels[0]?.id;
  const defaultLLMId = llmModels.find((m) => m.default)?.id ?? llmModels[0]?.id;

  useEffect(() => {
    if (defaultEmbedId && !form.getValues("embedding_model")) {
      form.setValue("embedding_model", defaultEmbedId);
    }
  }, [defaultEmbedId, form]);

  useEffect(() => {
    if (defaultLLMId && !form.getValues("llm_model")) {
      form.setValue("llm_model", defaultLLMId);
    }
  }, [defaultLLMId, form]);

  function handleSubmit(values: FormValues) {
    createClient.mutate(
      {
        name: values.name,
        description: values.description,
        embedding_model: values.embedding_model || undefined,
        llm_model: values.llm_model || undefined,
      },
      { onSuccess: () => form.reset() },
    );
  }

  const hasEmbedOptions = embedOptions.length > 0;
  const hasLLMOptions = llmOptions.length > 0;
  const colCount = 2 + (hasEmbedOptions ? 1 : 0) + (hasLLMOptions ? 1 : 0) + 1;

  return (
    <form
      className={`grid gap-4 lg:items-end ${colCount >= 5 ? "lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,1fr)_auto]" : "lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)_minmax(0,1fr)_auto]"}`}
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
      {hasEmbedOptions && (
        <div className="text-xs text-muted-foreground">
          Embedding model
          <DarkSelect
            label="Embedding model"
            className="mt-1"
            value={selectedEmbedModel}
            options={embedOptions}
            onChange={(value) => form.setValue("embedding_model", value)}
          />
        </div>
      )}
      {hasLLMOptions && (
        <div className="text-xs text-muted-foreground">
          LLM model
          <DarkSelect
            label="LLM model"
            className="mt-1"
            value={selectedLLMModel}
            options={llmOptions}
            onChange={(value) => form.setValue("llm_model", value)}
          />
        </div>
      )}
      <button className="button-primary mb-0 w-full lg:w-auto" disabled={createClient.isPending}>
        <Plus className="h-4 w-4" />
        Create workspace
      </button>
      {createClient.error && (
        <div className="lg:col-span-full">
          <ErrorState error={createClient.error} title="Create failed" />
        </div>
      )}
    </form>
  );
}
