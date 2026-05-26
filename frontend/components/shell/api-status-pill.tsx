import { StatusBadge } from "@/components/common/status-badge";

export function ApiStatusPill({ status = "degraded" }: { status?: string }) {
  return <StatusBadge status={status} label={`API ${status}`} />;
}
