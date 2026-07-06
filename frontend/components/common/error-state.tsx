"use client";

import { AlertCircle } from "lucide-react";
import { ApiError } from "@/lib/api/errors";

export function ErrorState({ error, title = "Request failed" }: { error: unknown; title?: string }) {
  const apiError = error instanceof ApiError ? error : null;
  const message = error instanceof Error ? error.message : String(error);

  return (
    <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4">
      <div className="flex gap-3">
        <AlertCircle className="mt-0.5 h-4 w-4 text-red-300" />
        <div>
          <h3 className="text-sm font-medium text-red-200">{title}</h3>
          <p className="mt-1 text-sm text-red-100/80">{message}</p>
          {apiError && (
            <details className="mt-3 text-xs text-red-100/70">
              <summary className="cursor-pointer">Technical detail</summary>
              <pre className="mt-2 max-h-64 overflow-auto rounded-md bg-background/80 p-3 font-mono">
                {JSON.stringify(
                  {
                    path: apiError.path,
                    status: apiError.status,
                    body: apiError.body,
                    validation: apiError.validation?.issues,
                  },
                  null,
                  2,
                )}
              </pre>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}
