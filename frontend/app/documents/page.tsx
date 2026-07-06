"use client";

import * as Tabs from "@radix-ui/react-tabs";
import { FileText, RotateCcw, Upload, X } from "lucide-react";
import { useEffect, useState } from "react";
import { useClients } from "@/lib/hooks/use-clients";
import { useDeleteDocument, useDocuments, useParsers, useRetryDocument, useUploadDocument } from "@/lib/hooks/use-documents";
import type { ParserInfo } from "@/lib/api/schemas";
import { useIngestionJobs, useRetryIngestionJob } from "@/lib/hooks/use-ingestion-jobs";
import { useWorkspaceStore } from "@/lib/state/workspace-store";
import { resolveEmbedLabel, useEmbeddingModels, resolveLLMLabel, useLLMModels } from "@/lib/hooks/use-models";
import { PageHeader } from "@/components/shell/page-header";
import { DocumentTable } from "@/components/documents/document-table";
import { IngestionJobCard } from "@/components/documents/ingestion-job-card";
import { EmptyState } from "@/components/common/empty-state";
import { ErrorState } from "@/components/common/error-state";
import { LoadingState } from "@/components/common/loading-state";
import { SectionCard } from "@/components/common/section-card";
import { DarkSelect } from "@/components/common/dark-select";
import { cn } from "@/lib/utils";

const defaultParserOptions: ParserInfo[] = [
  { id: "auto", label: "Auto (recommended)", available: true, description: "" },
  { id: "legacy", label: "Legacy", available: true, description: "" },
];

const STATUS_FILTER_OPTIONS = [
  { value: "all", label: "All statuses" },
  { value: "indexed", label: "Indexed" },
  { value: "processing", label: "Processing" },
  { value: "queued", label: "Queued" },
  { value: "failed", label: "Failed" },
];

const VALID_TABS = ["upload", "activity", "library"] as const;
type TabValue = (typeof VALID_TABS)[number];

export default function DocumentsPage() {
  const clients = useClients();
  const { workspaceId, setWorkspaceId } = useWorkspaceStore();
  const [requestedClientId, setRequestedClientId] = useState<string | null | undefined>(undefined);
  const storedClientId = clients.data?.some((client) => client.id === workspaceId) ? workspaceId : "";
  const validatedClientId = requestedClientId === undefined || !clients.data
    ? ""
    : requestedClientId && clients.data.some((client) => client.id === requestedClientId)
      ? requestedClientId
      : storedClientId || clients.data[0]?.id || "";
  const documents = useDocuments(validatedClientId);
  const jobs = useIngestionJobs(validatedClientId);
  const upload = useUploadDocument(validatedClientId);
  const retryDoc = useRetryDocument(validatedClientId);
  const deleteDoc = useDeleteDocument(validatedClientId);
  const retryJob = useRetryIngestionJob(validatedClientId);
  const parsers = useParsers();
  const { data: embeddingModels = [] } = useEmbeddingModels();
  const { data: llmModels = [] } = useLLMModels();

  const [tab, setTab] = useState<TabValue>("upload");
  const [files, setFiles] = useState<File[]>([]);
  const [globalParser, setGlobalParser] = useState("auto");
  const [fileParserMap, setFileParserMap] = useState<Record<string, string>>({});
  const [statusFilter, setStatusFilter] = useState("all");

  const activeJobs = (jobs.data ?? []).filter(
    (job) => job.status === "queued" || job.status === "processing" || job.status === "failed",
  );
  const filteredDocuments =
    statusFilter === "all"
      ? (documents.data ?? [])
      : (documents.data ?? []).filter((d) => d.status === statusFilter);

  const activeClient = clients.data?.find((c) => c.id === validatedClientId);
  const embedLabel = resolveEmbedLabel(activeClient?.embedding_model ?? null, embeddingModels);
  const llmLabel = resolveLLMLabel(activeClient?.llm_model ?? null, llmModels);

  // Resolve URL state before enabling workspace-scoped document queries.
  useEffect(() => {
    const hash = window.location.hash.replace("#", "") as TabValue;
    if (VALID_TABS.includes(hash)) setTab(hash);
    setRequestedClientId(new URLSearchParams(window.location.search).get("client_id"));
  }, []);

  useEffect(() => {
    if (requestedClientId === undefined || !clients.data) return;

    const requestedClient = requestedClientId
      ? clients.data.find((client) => client.id === requestedClientId)
      : undefined;
    if (requestedClient) {
      if (workspaceId !== requestedClient.id) setWorkspaceId(requestedClient.id);
      return;
    }

    const storedClientIsValid = clients.data.some((client) => client.id === workspaceId);
    if (!storedClientIsValid && clients.data[0]) setWorkspaceId(clients.data[0].id);
  }, [clients.data, requestedClientId, setWorkspaceId, workspaceId]);

  function handleTabChange(value: string) {
    setTab(value as TabValue);
    const url = new URL(window.location.href);
    url.hash = value;
    window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
  }

  function handleWorkspaceChange(clientId: string) {
    setWorkspaceId(clientId);
    setRequestedClientId(clientId || null);

    const url = new URL(window.location.href);
    if (clientId) {
      url.searchParams.set("client_id", clientId);
    } else {
      url.searchParams.delete("client_id");
    }
    window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
  }

  function fileKey(file: File, index: number) {
    return `${file.name}-${file.size}-${index}`;
  }

  function handleGlobalParserChange(value: string) {
    setGlobalParser(value);
    setFileParserMap({});
  }

  function setFileParser(file: File, index: number, value: string) {
    setFileParserMap((m) => ({ ...m, [fileKey(file, index)]: value }));
  }

  async function queueUpload() {
    if (!files.length) return;
    await Promise.all(
      files.map((file, index) => {
        const effective = fileParserMap[fileKey(file, index)] ?? globalParser;
        const parserPreference = effective === "auto" ? undefined : effective;
        return upload.mutateAsync({ file, parserPreference });
      }),
    );
    setFiles([]);
    setFileParserMap({});
    await documents.refetch();
    await jobs.refetch();
  }

  function addFiles(nextFiles: FileList | null) {
    if (!nextFiles?.length) return;
    setFiles((current) => [...current, ...Array.from(nextFiles)]);
  }

  function removeFile(index: number) {
    const key = fileKey(files[index], index);
    setFileParserMap((m) => { const n = { ...m }; delete n[key]; return n; });
    setFiles((current) => current.filter((_, i) => i !== index));
  }

  return (
    <div>
      <PageHeader
        title="Source material."
        description="Upload, ingest, version, inspect, retry, and delete documents within the active workspace."
        actions={
          <DarkSelect
            label="Workspace"
            value={validatedClientId}
            placeholder="Select workspace"
            onChange={handleWorkspaceChange}
            className="w-60"
            buttonClassName="font-mono"
            options={[
              { value: "", label: "Select workspace" },
              ...(clients.data ?? []).map((client) => ({ value: client.id, label: client.name })),
            ]}
          />
        }
      />
      <Tabs.Root value={tab} onValueChange={handleTabChange} className="space-y-4">
        <Tabs.List className="flex gap-1 border-b border-border">
          {VALID_TABS.map((t) => (
            <Tabs.Trigger key={t} value={t} className="border-b-2 border-transparent px-3 py-2 text-sm capitalize text-muted-foreground data-[state=active]:border-primary data-[state=active]:text-foreground">
              {t}
            </Tabs.Trigger>
          ))}
        </Tabs.List>

        {/* ── Upload ────────────────────────────────────────────────────── */}
        <Tabs.Content value="upload">
          <SectionCard title="Upload files" description="Supported formats: PDF, DOCX, PPTX. Max size: 200MB per file.">
            <div className="mb-4 flex items-start justify-between gap-4">
              <DarkSelect
                label="Parser (all files)"
                value={globalParser}
                onChange={handleGlobalParserChange}
                className="w-72"
                options={(parsers.data?.parsers ?? defaultParserOptions).map((p: ParserInfo) => ({
                  value: p.id,
                  label: p.label,
                  disabled: !p.available,
                  hint: p.available ? undefined : "not configured",
                }))}
              />
              {validatedClientId && embedLabel && (
                <span className="rounded-full border border-border bg-muted/40 px-3 py-1 text-xs text-muted-foreground">
                  {embedLabel}
                </span>
              )}
              {validatedClientId && llmLabel && (
                <span className="rounded-full border border-border bg-muted/40 px-3 py-1 text-xs text-muted-foreground">
                  {llmLabel}
                </span>
              )}
            </div>
            <div className="space-y-3">
              {files.length === 0 ? (
                <label
                  className="flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-border bg-muted/20 p-8 text-center hover:bg-muted/30"
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => { event.preventDefault(); addFiles(event.dataTransfer.files); }}
                >
                  <Upload className="h-8 w-8 text-muted-foreground" />
                  <span className="mt-3 text-sm font-medium">Drop source files here</span>
                  <span className="mt-1 text-xs text-muted-foreground">or browse from disk</span>
                  <input className="hidden" type="file" multiple accept=".pdf,.docx,.pptx" onChange={(event) => addFiles(event.target.files)} />
                </label>
              ) : (
                <>
                  <div className="rounded-md border border-border bg-background">
                    <div className="border-b border-border px-3 py-2 text-xs font-medium text-muted-foreground">
                      Selected files // {files.length}
                    </div>
                    <div className="max-h-[28rem] divide-y divide-border overflow-y-auto">
                      {files.map((file, index) => {
                        const key = fileKey(file, index);
                        const effectiveParser = fileParserMap[key] ?? globalParser;
                        const parserOptions = (parsers.data?.parsers ?? defaultParserOptions).map((p: ParserInfo) => ({
                          value: p.id,
                          label: p.label,
                          disabled: !p.available,
                          hint: p.available ? undefined : "not configured",
                        }));
                        return (
                          <div key={key} className="flex items-start gap-3 px-4 py-3">
                            <FileText className="mt-1 h-4 w-4 shrink-0 text-muted-foreground" />
                            <div className="min-w-0 flex-1">
                              <div className="truncate text-sm font-medium">{file.name}</div>
                              <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[11px] text-muted-foreground">
                                <span>{formatFileSize(file.size)}</span>
                                <span>{file.type || fileExtension(file.name) || "unknown type"}</span>
                              </div>
                              <div className="mt-2">
                                <DarkSelect
                                  label={`Parser for ${file.name}`}
                                  value={effectiveParser}
                                  onChange={(value) => setFileParser(file, index, value)}
                                  className="w-52"
                                  options={parserOptions}
                                />
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
                        );
                      })}
                    </div>
                  </div>
                  <label
                    className="flex h-14 cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-border bg-muted/20 text-sm text-muted-foreground hover:bg-muted/30"
                    onDragOver={(event) => event.preventDefault()}
                    onDrop={(event) => { event.preventDefault(); addFiles(event.dataTransfer.files); }}
                  >
                    <Upload className="h-4 w-4" />
                    <span>Drop more files or browse from disk</span>
                    <input className="hidden" type="file" multiple accept=".pdf,.docx,.pptx" onChange={(event) => addFiles(event.target.files)} />
                  </label>
                </>
              )}
              <button className="button-primary w-full" disabled={!workspaceId || !files.length || upload.isPending} onClick={queueUpload}>
                Queue ingestion
              </button>
              {upload.error && <ErrorState error={upload.error} title="Upload failed" />}
            </div>
          </SectionCard>
        </Tabs.Content>

        {/* ── Activity ──────────────────────────────────────────────────── */}
        <Tabs.Content value="activity">
          <div className="mb-3 flex justify-end">
            <button
              className="button-secondary"
              onClick={() => jobs.refetch()}
              disabled={jobs.isFetching}
            >
              <RotateCcw className={cn("h-4 w-4", jobs.isFetching && "animate-spin")} />
              Refresh
            </button>
          </div>
          {jobs.isLoading ? <LoadingState /> : jobs.error ? <ErrorState error={jobs.error} /> : activeJobs.length ? (
            <>
              {activeJobs.some((j) => j.status === "queued" || j.status === "processing") && (
                <div className="mb-3 rounded-md border border-border bg-muted/30 px-4 py-3 text-xs text-muted-foreground">
                  Parsing may take several minutes depending on document size and parser. Use{" "}
                  <span className="font-medium text-foreground">Refresh</span> to check for updates.
                </div>
              )}
              <div className="space-y-3">
                {activeJobs.map((job) => (
                  <IngestionJobCard
                    key={job.id}
                    job={job}
                    onRetry={(parser) => retryJob.mutate({ jobId: job.id, parser })}
                    retrying={retryJob.isPending}
                  />
                ))}
              </div>
            </>
          ) : <EmptyState title="No active ingestion jobs" description="Queued, processing, and failed ingestion jobs will appear here." />}
        </Tabs.Content>

        {/* ── Library ───────────────────────────────────────────────────── */}
        <Tabs.Content value="library">
          {documents.isLoading ? <LoadingState /> : documents.error ? <ErrorState error={documents.error} /> : documents.data?.length ? (
            <SectionCard
              title={`Document library // ${documents.data.length}`}
              actions={
                <DarkSelect
                  label="Filter by status"
                  value={statusFilter}
                  onChange={setStatusFilter}
                  options={STATUS_FILTER_OPTIONS}
                  className="w-44"
                  menuSide="bottom"
                />
              }
            >
              {filteredDocuments.length === 0 ? (
                <p className="py-8 text-center text-sm text-muted-foreground">
                  No documents match the selected filter.
                </p>
              ) : (
                <DocumentTable
                  documents={filteredDocuments}
                  onDelete={(id) => deleteDoc.mutate(id)}
                  onRetry={(id) => retryDoc.mutate(id)}
                  pending={deleteDoc.isPending || retryDoc.isPending}
                />
              )}
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
