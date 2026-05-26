import { SectionCard } from "@/components/common/section-card";
import type { RetrievalTrace } from "@/lib/api/schemas";
import { scoreLabel } from "@/lib/utils";

export function RetrievalTraceView({ retrieval }: { retrieval: RetrievalTrace }) {
  const rows = [
    ["Mode", retrieval.mode],
    ["Dense score", scoreLabel(retrieval.dense_score)],
    ["Sparse score", scoreLabel(retrieval.sparse_score)],
    ["Rerank score", scoreLabel(retrieval.rerank_score)],
    ["Top K", retrieval.top_k ?? "-"],
    ["Sparse available", retrieval.sparse_available === undefined ? "-" : retrieval.sparse_available ? "yes" : "no"],
    ["Fallback reason", retrieval.fallback_reason ?? "-"],
  ];

  return (
    <SectionCard>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
        {rows.map(([label, value]) => (
          <div key={String(label)}>
            <dt className="text-xs text-muted-foreground">{label}</dt>
            <dd className="mt-1 font-mono text-xs text-foreground">{String(value)}</dd>
          </div>
        ))}
      </dl>
    </SectionCard>
  );
}
