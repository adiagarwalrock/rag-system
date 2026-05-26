import { FileText } from "lucide-react";
import type { Citation } from "@/lib/api/schemas";
import { scoreLabel } from "@/lib/utils";

export function CitationChip({ citation, index }: { citation: Citation; index: number }) {
  return (
    <span className="inline-flex max-w-full items-center gap-1.5 rounded-md border border-border bg-muted px-2 py-1 text-xs text-muted-foreground">
      <FileText className="h-3.5 w-3.5 shrink-0" />
      <span className="truncate">{index + 1}. {citation.filename}</span>
      {citation.page && <span className="font-mono">p.{citation.page}</span>}
      {citation.score !== undefined && <span className="font-mono">{scoreLabel(citation.score)}</span>}
    </span>
  );
}
