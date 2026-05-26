"use client";

import * as Tabs from "@radix-ui/react-tabs";
import { Upload } from "lucide-react";
import { useEffect, useState } from "react";
import { useClients } from "@/lib/hooks/use-clients";
import { useDeleteDocument, useDocuments, useRetryDocument, useUploadDocument } from "@/lib/hooks/use-documents";
import { useIngestionJobs, useRetryIngestionJob } from "@/lib/hooks/use-ingestion-jobs";
import { useWorkspaceStore } from "@/lib/state/workspace-store";
import { PageHeader } from "@/components/shell/page-header";
import { DocumentTable } from "@/components/documents/document-table";
import { IngestionJobCard } from "@/components/documents/ingestion-job-card";
import { EmptyState } from "@/components/common/empty-state";
import { ErrorState } from "@/components/common/error-state";
import { LoadingState } from "@/components/common/loading-state";
import { SectionCard } from "@/components/common/section-card";

export default function DocumentsPage() {
  const clients = useClients();
  const { workspaceId, setWorkspaceId } = useWorkspaceStore();
  const documents = useDocuments(workspaceId);
  const jobs = useIngestionJobs(workspaceId);
  const upload = useUploadDocument(workspaceId);
  const retryDoc = useRetryDocument(workspaceId);
  const deleteDoc = useDeleteDocument(workspaceId);
  const retryJob = useRetryIngestionJob(workspaceId);
  const [files, setFiles] = useState<FileList | null>(null);
  const [parser, setParser] = useState("auto fallback");
  const [versionMode, setVersionMode] = useState("new document family");
  const [family, setFamily] = useState("");

  useEffect(() => {
    if (!workspaceId && clients.data?.[0]) setWorkspaceId(clients.data[0].id);
  }, [clients.data, setWorkspaceId, workspaceId]);

  async function queueUpload() {
    if (!files) return;
    await Promise.all(
      Array.from(files).map((file) => upload.mutateAsync({ file, parserPreference: parser })),
    );
    await documents.refetch();
    await jobs.refetch();
  }

  return (
    <div>
      <PageHeader
        title="Source material."
        description="Upload, ingest, version, inspect, retry, and delete documents within the active workspace."
        actions={
          <select className="control w-60 font-mono text-xs" value={workspaceId} onChange={(event) => setWorkspaceId(event.target.value)}>
            <option value="">Select workspace</option>
            {(clients.data ?? []).map((client) => <option key={client.id} value={client.id}>{client.name}</option>)}
          </select>
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
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
              <label className="flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-border bg-muted/20 p-8 text-center hover:bg-muted/30">
                <Upload className="h-8 w-8 text-muted-foreground" />
                <span className="mt-3 text-sm font-medium">Drop source files here</span>
                <span className="mt-1 text-xs text-muted-foreground">or browse from disk</span>
                <input className="hidden" type="file" multiple accept=".pdf,.docx,.pptx" onChange={(event) => setFiles(event.target.files)} />
              </label>
              <div className="space-y-3">
                <label className="block text-xs text-muted-foreground">
                  Parser preference
                  <select className="control mt-1 w-full" value={parser} onChange={(event) => setParser(event.target.value)}>
                    {["auto fallback", "reducto", "llamaparse", "layout_pdf", "legacy"].map((value) => <option key={value}>{value}</option>)}
                  </select>
                </label>
                <label className="block text-xs text-muted-foreground">
                  Version mode
                  <select className="control mt-1 w-full" value={versionMode} onChange={(event) => setVersionMode(event.target.value)}>
                    <option>new document family</option>
                    <option>new version of existing document</option>
                  </select>
                </label>
                {versionMode.includes("existing") && (
                  <label className="block text-xs text-muted-foreground">
                    Existing document family
                    <select className="control mt-1 w-full font-mono text-xs" value={family} onChange={(event) => setFamily(event.target.value)}>
                      <option value="">Select family</option>
                      {(documents.data ?? []).map((document) => <option key={document.id} value={document.document_family}>{document.document_family}</option>)}
                    </select>
                  </label>
                )}
                <div className="rounded-md border border-border bg-background p-3 text-xs text-muted-foreground">
                  {files?.length ? `${files.length} file(s) selected.` : "No files selected."}
                </div>
                <button className="button-primary w-full" disabled={!workspaceId || !files?.length || upload.isPending} onClick={queueUpload}>
                  Queue ingestion
                </button>
                {upload.error && <ErrorState error={upload.error} title="Upload failed" />}
              </div>
            </div>
          </SectionCard>
        </Tabs.Content>
        <Tabs.Content value="activity">
          {jobs.isLoading ? <LoadingState /> : jobs.error ? <ErrorState error={jobs.error} /> : jobs.data?.length ? (
            <div className="space-y-3">
              {jobs.data.map((job) => <IngestionJobCard key={job.id} job={job} onRetry={() => retryJob.mutate(job.id)} retrying={retryJob.isPending} />)}
            </div>
          ) : <EmptyState title="No ingestion jobs" description="Queued and processing documents will appear here." />}
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
