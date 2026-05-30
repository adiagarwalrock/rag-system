import { SectionCard } from "@/components/common/section-card";
import type { RetrievalTrace } from "@/lib/api/schemas";

export function RetrievalTraceView({ retrieval }: { retrieval: RetrievalTrace }) {
  const rows = [
    ["Mode", retrieval.mode],
    ["Top K", retrieval.top_k ?? "-"],
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
