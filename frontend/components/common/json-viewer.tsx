import { cn } from "@/lib/utils";

export function JsonViewer({
  value,
  wrap = false,
  fullHeight = false,
}: {
  value: unknown;
  wrap?: boolean;
  fullHeight?: boolean;
}) {
  return (
    <pre
      className={cn(
        "overflow-auto rounded-md border border-border bg-background p-3 font-mono text-xs leading-5 text-muted-foreground",
        fullHeight ? "h-full max-h-none" : "max-h-[520px]",
        wrap && "whitespace-pre-wrap break-words",
      )}
    >
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}
