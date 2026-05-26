"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { EmptyState } from "@/components/common/empty-state";
import { ErrorState } from "@/components/common/error-state";
import { LoadingState } from "@/components/common/loading-state";
import { SectionCard } from "@/components/common/section-card";
import { DarkSelect } from "@/components/common/dark-select";
import { QualityEvalPanel } from "@/components/quality/quality-eval-panel";
import { PageHeader } from "@/components/shell/page-header";
import { useClients } from "@/lib/hooks/use-clients";
import { useCreateQualityTest, useQualityRuns, useQualityTests, useRunQualityTest } from "@/lib/hooks/use-quality";
import { useWorkspaceStore } from "@/lib/state/workspace-store";

const schema = z.object({
  name: z.string().min(2),
  question: z.string().min(4),
  expected_answer: z.string().optional(),
  required_citations: z.string().optional(),
  expected_conflict: z.boolean().optional(),
});

export default function QualityPage() {
  const clients = useClients();
  const { workspaceId, setWorkspaceId } = useWorkspaceStore();
  const tests = useQualityTests(workspaceId);
  const runs = useQualityRuns(workspaceId);
  const createTest = useCreateQualityTest(workspaceId);
  const runTest = useRunQualityTest(workspaceId);
  const form = useForm<z.infer<typeof schema>>({ resolver: zodResolver(schema), defaultValues: { name: "", question: "", expected_answer: "", required_citations: "", expected_conflict: false } });

  useEffect(() => {
    if (!workspaceId && clients.data?.[0]) setWorkspaceId(clients.data[0].id);
  }, [clients.data, setWorkspaceId, workspaceId]);

  return (
    <div>
      <PageHeader
        title="Quality evaluation."
        description="Measure ingestion quality, retrieval recall, citation accuracy, and answer faithfulness."
        actions={
          <DarkSelect
            label="Workspace"
            value={workspaceId}
            placeholder="Select workspace"
            onChange={setWorkspaceId}
            className="w-60"
            buttonClassName="font-mono"
            options={[
              { value: "", label: "Select workspace" },
              ...(clients.data ?? []).map((client) => ({ value: client.id, label: client.name })),
            ]}
          />
        }
      />
      <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
        <SectionCard title="Test case">
          <form className="space-y-3" onSubmit={form.handleSubmit((values) => createTest.mutate({
            name: values.name,
            question: values.question,
            expected_answer: values.expected_answer || undefined,
            required_citations: values.required_citations?.split(",").map((item) => item.trim()).filter(Boolean),
            expected_conflict: values.expected_conflict,
          }, { onSuccess: () => form.reset() }))}>
            <input className="control w-full" placeholder="test name" {...form.register("name")} />
            <textarea className="min-h-24 w-full rounded-md border border-border bg-muted p-3 text-sm outline-none focus:border-primary" placeholder="question" {...form.register("question")} />
            <textarea className="min-h-24 w-full rounded-md border border-border bg-muted p-3 text-sm outline-none focus:border-primary" placeholder="expected answer" {...form.register("expected_answer")} />
            <input className="control w-full" placeholder="required citations, comma separated" {...form.register("required_citations")} />
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" {...form.register("expected_conflict")} /> Expected conflict</label>
            <button className="button-primary w-full" disabled={!workspaceId || createTest.isPending}>Create test case</button>
            {createTest.error && <ErrorState error={createTest.error} title="Create failed" />}
          </form>
        </SectionCard>
        <div>
          {tests.isLoading || runs.isLoading ? <LoadingState /> : tests.error ? <ErrorState error={tests.error} /> : tests.data?.length ? <QualityEvalPanel tests={tests.data} runs={runs.data ?? []} onRun={(id) => runTest.mutate(id)} running={runTest.isPending} /> : <EmptyState title="No quality tests" description="Create a test case to evaluate retrieval and answer quality." />}
        </div>
      </div>
    </div>
  );
}
