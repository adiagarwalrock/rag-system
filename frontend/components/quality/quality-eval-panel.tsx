"use client";

import { Play } from "lucide-react";
import type { QualityRun, QualityTest } from "@/lib/api/schemas";
import { JsonViewer } from "@/components/common/json-viewer";
import { StatusBadge } from "@/components/common/status-badge";

export function QualityEvalPanel({
  tests,
  runs,
  onRun,
  running,
}: {
  tests: QualityTest[];
  runs: QualityRun[];
  onRun: (testId: string) => void;
  running?: boolean;
}) {
  const latestByTest = new Map<string, QualityRun>();
  for (const run of runs) {
    if (!latestByTest.has(run.test_id)) latestByTest.set(run.test_id, run);
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
      <div className="overflow-x-auto rounded-lg border border-border bg-card">
        <table className="w-full min-w-[900px] text-left text-sm">
          <thead className="border-b border-border text-xs text-muted-foreground">
            <tr>
              <th className="p-3 font-medium">Test name</th>
              <th className="p-3 font-medium">Status</th>
              <th className="p-3 font-medium">Retrieval recall</th>
              <th className="p-3 font-medium">Citation match</th>
              <th className="p-3 font-medium">Faithfulness</th>
              <th className="p-3 font-medium">Conflict</th>
              <th className="p-3 font-medium">Latency</th>
              <th className="p-3 font-medium">Run</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {tests.map((test) => {
              const run = latestByTest.get(test.id);
              return (
                <tr key={test.id}>
                  <td className="max-w-xs p-3">
                    <div className="font-medium">{test.name}</div>
                    <div className="mt-1 truncate text-xs text-muted-foreground">{test.question}</div>
                  </td>
                  <td className="p-3"><StatusBadge status={run?.status === "passed" ? "ok" : run?.status === "failed" ? "error" : "degraded"} label={run?.status ?? "not run"} /></td>
                  <td className="p-3 font-mono text-xs">{score(run?.retrieval_recall)}</td>
                  <td className="p-3 font-mono text-xs">{score(run?.citation_match)}</td>
                  <td className="p-3 font-mono text-xs">{score(run?.answer_faithfulness)}</td>
                  <td className="p-3">{run?.conflict_detection === undefined ? "-" : run.conflict_detection ? "yes" : "no"}</td>
                  <td className="p-3 font-mono text-xs">{run?.latency_ms ? `${run.latency_ms} ms` : "-"}</td>
                  <td className="p-3"><button className="button-secondary" onClick={() => onRun(test.id)} disabled={running}><Play className="h-4 w-4" />Run</button></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="rounded-lg border border-border bg-card p-4">
        <h2 className="text-sm font-semibold">Result panel</h2>
        <p className="mt-1 text-xs text-muted-foreground">Latest evaluator output, including missing citations, unexpected citations, conflicting evidence, notes, and raw JSON.</p>
        <div className="mt-4">
          <JsonViewer value={runs[0] ?? { status: "no runs yet" }} />
        </div>
      </div>
    </div>
  );
}

function score(value?: number) {
  return typeof value === "number" ? value.toFixed(2) : "-";
}
