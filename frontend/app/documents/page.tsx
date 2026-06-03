"use client";

import * as Tabs from "@radix-ui/react-tabs";
import { FileText, Upload, X } from "lucide-react";
import { useEffect, useState } from "react";
import { useClients } from "@/lib/hooks/use-clients";
import { useDeleteDocument, useDocuments, useParsers, useRetryDocument, useUploadDocument } from "@/lib/hooks/use-documents";
import type { ParserInfo } from "@/lib/api/schemas";
import { useIngestionJobs, useRetryIngestionJob } from "@/lib/hooks/use-ingestion-jobs";
import { useWorkspaceStore } from "@/lib/state/workspace-store";
import { PageHeader } from "@/components/shell/page-header";
import { DocumentTable } from "@/components/documents/document-table";
import { IngestionJobCard } from "@/components/documents/ingestion-job-card";
import { EmptyState } from "@/components/common/empty-state";
import { ErrorState } from "@/components/common/error-state";
import { LoadingState } from "@/components/common/loading-state";
import { SectionCard } from "@/components/common/section-card";
import { DarkSelect } from "@/components/common/dark-select";

const defaultParserOptions: ParserInfo[] = [
  { id: "auto", label: "Auto (recommended)", available: true, description: "" },
  { id: "legacy", label: "Legacy", available: true, description: "" },
];

export default function DocumentsPage() {
  const clients = useClients();
  const { workspaceId, setWorkspaceId } = useWorkspaceStore();
  const documents = useDocuments(workspaceId);
  const jobs = useIngestionJobs(workspaceId);
  const upload = useUploadDocument(workspaceId);
  const retryDoc = useRetryDocument(workspaceId);
  const deleteDoc = useDeleteDocument(workspaceId);
  const retryJob = useRetryIngestionJob(workspaceId);
  const parsers = useParsers();
  const [files, setFiles] = useState<File[]>([]);
  const [selectedParser, setSelectedParser] = useState("auto");
  const activeJobs = (jobs.data ?? []).filter((job) =>
    job.status === "queued" || job.status === "processing" || job.status === "failed",
  );

  useEffect(() => {
    if (!workspaceId && clients.data?.[0]) setWorkspaceId(clients.data[0].id);
  }, [clients.data, setWorkspaceId, workspaceId]);

  async function queueUpload() {
    if (!files.length) return;
    const parserPreference = selectedParser === "auto" ? undefined : selectedParser;
    await Promise.all(
      files.map((file) => upload.mutateAsync({ file, parserPreference })),
    );
    setFiles([]);
    await documents.refetch();
    await jobs.refetch();
  }

  function addFiles(nextFiles: FileList | null) {
    if (!nextFiles?.length) return;
    setFiles((current) => [...current, ...Array.from(nextFiles)]);
  }

  function removeFile(index: number) {
    setFiles((current) => current.filter((_, currentIndex) => currentIndex !== index));
  }

  return (
    <div>
      <PageHeader
        title="Source material."
        description="Upload, ingest, version, inspect, retry, and delete documents within the active workspace."
        actions={
          <DarkSelect
            label="Workspace"
            value={workspaceId}
            placeholder="Select workspace"
            onChange={setWorkspaceId}
            className="w-60"
            buttonClassName="font-mono"
            options={[
              { value: "", label: "Select workspace" },
              ...(clients.data ?? []).map((client) => ({ value: client.id, label: client.name })),
            ]}
          />
        }
      />
      <Tabs.Root defaultValue="upload" className="space-y-4">
        <Tabs.List className="flex gap-1 border-b border-border">
          {["upload", "activity", "library"].map((tab) => (
            <Tabs.Trigger key={tab} value={tab} className="border-b-2 border-transparent px-3 py-2 text-sm capitalize text-muted-foreground data-[state=active]:border-primary data-[state=active]:text-foreground">
              {tab}
            </Tabs.Trigger>
          ))}
        </Tabs.List>
        <Tabs.Content value="upload">
          <SectionCard title="Upload files" description="Supported formats: PDF, DOCX, PPTX. Max size: 200MB per file.">
            <div className="mb-4">
              <DarkSelect
                label="Parser"
                value={selectedParser}
                onChange={setSelectedParser}
                className="w-72"
                options={(parsers.data?.parsers ?? defaultParserOptions).map((p: ParserInfo) => ({
                  value: p.id,
                  label: p.label,
                  disabled: !p.available,
                  hint: p.available ? undefined : "not configured",
                }))}
              />
            </div>
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
              <label
                className="flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-border bg-muted/20 p-8 text-center hover:bg-muted/30"
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  addFiles(event.dataTransfer.files);
                }}
              >
                <Upload className="h-8 w-8 text-muted-foreground" />
                <span className="mt-3 text-sm font-medium">Drop source files here</span>
                <span className="mt-1 text-xs text-muted-foreground">or browse from disk</span>
                <input className="hidden" type="file" multiple accept=".pdf,.docx,.pptx" onChange={(event) => addFiles(event.target.files)} />
              </label>
              <div className="space-y-3">
                <div className="rounded-md border border-border bg-background">
                  <div className="border-b border-border px-3 py-2 text-xs font-medium text-muted-foreground">
                    Selected files
                  </div>
                  {files.length ? (
                    <div className="max-h-56 divide-y divide-border overflow-y-auto">
                      {files.map((file, index) => (
                        <div key={`${file.name}-${file.size}-${index}`} className="flex items-start gap-2 px-3 py-2">
                          <FileText className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-sm font-medium">{file.name}</div>
                            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[11px] text-muted-foreground">
                              <span>{formatFileSize(file.size)}</span>
                              <span>{file.type || fileExtension(file.name) || "unknown type"}</span>
                            </div>
                          </div>
                          <button
                            type="button"
                            className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
                            onClick={() => removeFile(index)}
                            aria-label={`Remove ${file.name}`}
                          >
                            <X className="h-4 w-4" />
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="px-3 py-6 text-center text-xs text-muted-foreground">
                      No files selected.
                    </div>
                  )}
                </div>
                <button className="button-primary w-full" disabled={!workspaceId || !files.length || upload.isPending} onClick={queueUpload}>
                  Queue ingestion
                </button>
                {upload.error && <ErrorState error={upload.error} title="Upload failed" />}
              </div>
            </div>
          </SectionCard>
        </Tabs.Content>
        <Tabs.Content value="activity">
          {jobs.isLoading ? <LoadingState /> : jobs.error ? <ErrorState error={jobs.error} /> : activeJobs.length ? (
            <div className="space-y-3">
              {activeJobs.map((job) => <IngestionJobCard key={job.id} job={job} onRetry={() => retryJob.mutate(job.id)} retrying={retryJob.isPending} />)}
            </div>
          ) : <EmptyState title="No active ingestion jobs" description="Queued, processing, and failed ingestion jobs will appear here." />}
        </Tabs.Content>
        <Tabs.Content value="library">
          {documents.isLoading ? <LoadingState /> : documents.error ? <ErrorState error={documents.error} /> : documents.data?.length ? (
            <SectionCard title={`Document library // ${documents.data.length}`}>
              <DocumentTable
                documents={documents.data}
                onDelete={(id) => deleteDoc.mutate(id)}
                onRetry={(id) => retryDoc.mutate(id)}
                pending={deleteDoc.isPending || retryDoc.isPending}
              />
            </SectionCard>
          ) : <EmptyState title="No documents uploaded" description="Upload source files to make them available for scoped RAG queries." />}
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
}

function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

function fileExtension(filename: string) {
  const extension = filename.split(".").pop();
  return extension ? `.${extension.toLowerCase()}` : "";
}
