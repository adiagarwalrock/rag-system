import { AlertTriangle, CheckCircle2, CircleDashed, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

const styles = {
  ok: "border-success/30 bg-success/10 text-green-300",
  success: "border-success/30 bg-success/10 text-green-300",
  indexed: "border-success/30 bg-success/10 text-green-300",
  degraded: "border-warning/30 bg-warning/10 text-amber-300",
  warning: "border-warning/30 bg-warning/10 text-amber-300",
  queued: "border-blue-400/30 bg-blue-400/10 text-blue-300",
  processing: "border-blue-400/30 bg-blue-400/10 text-blue-300",
  error: "border-destructive/35 bg-destructive/10 text-red-300",
  failed: "border-destructive/35 bg-destructive/10 text-red-300",
  deleted: "border-border bg-muted text-muted-foreground",
  neutral: "border-border bg-muted text-muted-foreground",
};

const icons: Partial<Record<keyof typeof styles, React.ComponentType<{ className?: string }>>> = {
  ok: CheckCircle2,
  success: CheckCircle2,
  indexed: CheckCircle2,
  error: XCircle,
  failed: XCircle,
  queued: CircleDashed,
  processing: CircleDashed,
};

export function StatusBadge({
  status,
  label,
  className,
}: {
  status?: string;
  label?: string;
  className?: string;
}) {
  const key = (status ?? "neutral").toLowerCase() as keyof typeof styles;
  const Icon = icons[key] ?? AlertTriangle;

  return (
    <span
      className={cn(
        "inline-flex h-6 items-center gap-1.5 rounded-md border px-2 text-xs font-medium",
        styles[key] ?? styles.neutral,
        className,
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {label ?? status ?? "unknown"}
    </span>
  );
}
