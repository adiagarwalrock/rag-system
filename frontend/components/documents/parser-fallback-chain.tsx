import { ArrowRight } from "lucide-react";

const chain = ["Reducto API", "LlamaParse", "Layout-aware PDF", "Legacy parser"];

export function ParserFallbackChain() {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
      {chain.map((item, index) => (
        <span key={item} className="inline-flex items-center gap-2">
          <span className="rounded-md border border-border bg-muted px-2 py-1">{item}</span>
          {index < chain.length - 1 && <ArrowRight className="h-3.5 w-3.5" />}
        </span>
      ))}
    </div>
  );
}
