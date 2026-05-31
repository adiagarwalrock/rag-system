import { SectionCard } from "@/components/common/section-card";
import type { RetrievalTrace } from "@/lib/api/schemas";

export function RetrievalTraceView({ retrieval }: { retrieval: RetrievalTrace }) {
  const rows = [
    ["Mode", retrieval.mode],
    ["Query expanded", retrieval.query_expanded ? "yes" : "no"],
    ["Image referenced", retrieval.image_referenced ? "yes" : "no"],
    ["Evidence / sources", `${retrieval.evidence_count ?? "-"} / ${retrieval.source_count ?? "-"}`],
    ["Image chunks", `${retrieval.evidence_image_chunk_count ?? 0} / ${retrieval.ranked_image_chunk_count ?? 0}`],
    ["Images sent", retrieval.images_used_count ?? 0],
    ["Intent", retrieval.intent_labels.length ? retrieval.intent_labels.join(", ") : "-"],
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
