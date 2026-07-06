import type { RuntimeStatus } from "@/lib/api/schemas";
import { MetricCard } from "@/components/common/metric-card";
import { SectionCard } from "@/components/common/section-card";
import { StatusBadge } from "@/components/common/status-badge";
import { formatNumber } from "@/lib/utils";

export function RuntimeDiagnostics({ status }: { status: RuntimeStatus }) {
  const metadata = {
    database: status.database.metadata ?? {},
    qdrant: status.qdrant.metadata ?? {},
    ai: status.ai_provider.metadata ?? {},
  };
  const inUseCollections = getInUseCollections(metadata.qdrant);

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-5">
        <MetricCard label="Database" value={<StatusBadge status={status.database.status} />} />
        <MetricCard label="Qdrant" value={<StatusBadge status={status.qdrant.status} />} />
        <MetricCard label="AI provider" value={<StatusBadge status={status.ai_provider.status} />} />
        <MetricCard label="Parsers" value={status.parsers.length} detail="fallback chain configured" />
        <MetricCard label="API" value={<StatusBadge status={status.api?.status ?? "degraded"} />} />
      </div>

      <SectionCard title="Database">
        <InfoGrid items={metadata.database} preferred={["mode", "dialect", "target", "status"]} />
      </SectionCard>

      <SectionCard title="AI provider">
        <InfoGrid items={metadata.ai} preferred={["provider", "api_mode", "key_valid", "llm_model", "embedding_model", "embedding_dimensions"]} />
      </SectionCard>

      <SectionCard title="Parser fallback chain">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-sm">
            <thead className="border-b border-border text-xs text-muted-foreground">
              <tr>
                <th className="py-2 pr-3 font-medium">Parser</th>
                <th className="py-2 pr-3 font-medium">Configured</th>
                <th className="py-2 pr-3 font-medium">Enabled</th>
                <th className="py-2 pr-3 font-medium">Priority</th>
                <th className="py-2 pr-3 font-medium">Latest</th>
                <th className="py-2 pr-3 font-medium">Avg parse</th>
                <th className="py-2 font-medium">Error</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {status.parsers.map((parser) => (
                <tr key={parser.name}>
                  <td className="py-3 pr-3 font-mono text-xs">{parser.name}</td>
                  <td className="py-3 pr-3">{parser.configured ? "yes" : "no"}</td>
                  <td className="py-3 pr-3">{parser.enabled ? "yes" : "no"}</td>
                  <td className="py-3 pr-3 font-mono text-xs">{parser.priority}</td>
                  <td className="py-3 pr-3"><StatusBadge status={parser.latest_status === "unused" ? "neutral" : parser.latest_status} label={parser.latest_status} /></td>
                  <td className="py-3 pr-3 font-mono text-xs">{parser.average_parse_time_ms ?? "-"}</td>
                  <td className="py-3 text-xs text-muted-foreground">{parser.latest_error ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Qdrant collections">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-sm">
            <thead className="border-b border-border text-xs text-muted-foreground">
              <tr>
                <th className="py-2 pr-3 font-medium">Collection</th>
                <th className="py-2 pr-3 font-medium">Role</th>
                <th className="py-2 pr-3 font-medium">Points</th>
                <th className="py-2 pr-3 font-medium">Vector</th>
                <th className="py-2 pr-3 font-medium">Health</th>
                <th className="py-2 font-medium">Updated</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {status.collections.map((collection) => (
                <tr key={collection.name}>
                  <td className="py-3 pr-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-xs">{collection.name}</span>
                      {inUseCollections.has(collection.name) ? (
                        <StatusBadge status="ok" label="in-use" />
                      ) : null}
                    </div>
                  </td>
                  <td className="py-3 pr-3">{collection.role}</td>
                  <td className="py-3 pr-3 font-mono text-xs">{formatNumber(collection.point_count)}</td>
                  <td className="py-3 pr-3 font-mono text-xs">{collection.vector_type ?? "-"}</td>
                  <td className="py-3 pr-3"><StatusBadge status={collection.health} /></td>
                  <td className="py-3 font-mono text-xs text-muted-foreground">{collection.last_updated ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="REST API">
        <InfoGrid items={status.api?.metadata ?? {}} preferred={["base_url", "status"]} />
        <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
          {["clients", "documents", "query", "health", "qdrant", "quality"].map((domain) => (
            <span key={domain} className="rounded-md border border-border bg-muted px-2 py-1">{domain}</span>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function getInUseCollections(metadata: Record<string, unknown>) {
  const names = new Set<string>();
  for (const key of ["document_collection", "chat_history_collection"]) {
    const value = metadata[key];
    if (typeof value === "string" && value.trim()) names.add(value);
  }
  return names;
}

function InfoGrid({ items, preferred }: { items: Record<string, unknown>; preferred: string[] }) {
  const keys = preferred.filter((key) => key in items);
  if (!keys.length) {
    return <p className="text-sm text-muted-foreground">No structured metadata returned.</p>;
  }
  return (
    <dl className="grid gap-3 md:grid-cols-3">
      {keys.map((key) => (
        <div key={key} className="rounded-md border border-border bg-muted/30 p-3">
          <dt className="text-xs text-muted-foreground">{key}</dt>
          <dd className="mt-1 break-all font-mono text-xs">{JSON.stringify(items[key])}</dd>
        </div>
      ))}
    </dl>
  );
}
