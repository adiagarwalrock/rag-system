import { StatusBadge } from "@/components/common/status-badge";

export function RuntimeHealthPill({ status = "degraded" }: { status?: string }) {
  return <StatusBadge status={status} label={`Runtime ${status}`} />;
}
