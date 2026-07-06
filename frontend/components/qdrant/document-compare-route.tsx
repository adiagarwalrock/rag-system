"use client";

import Link from "next/link";
import { Download, RefreshCw, Search } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { DarkSelect } from "@/components/common/dark-select";
import { EmptyState } from "@/components/common/empty-state";
import { ErrorState } from "@/components/common/error-state";
import { LoadingState } from "@/components/common/loading-state";
import { SectionCard } from "@/components/common/section-card";
import { StatusBadge } from "@/components/common/status-badge";
import { DocumentComparison } from "@/components/qdrant/document-comparison";
import {
  DocumentSelectionTable,
  maxSelectedDocuments,
  selectedDocumentCountLabel,
} from "@/components/qdrant/document-selection-table";
import { facetOptions, mergeOptions, parserOptions } from "@/components/qdrant/node-picker-options";
import { PageHeader } from "@/components/shell/page-header";
import { useClients } from "@/lib/hooks/use-clients";
import { useDocuments, useParsers } from "@/lib/hooks/use-documents";
import {
  useQdrantCollections,
  useQdrantDocumentCompare,
  useQdrantDocuments,
} from "@/lib/hooks/use-qdrant";
import { downloadJson } from "@/lib/utils";

export function DocumentCompareRoute({
  collection,
  documentKeys,
}: {
  collection: string;
  documentKeys: string[];
}) {
  const router = useRouter();
  const normalizedInitialKeys = useMemo(
    () => normalizeDocumentKeys(documentKeys),
    [documentKeys],
  );
  const clients = useClients();
  const parsers = useParsers();
  const collections = useQdrantCollections();
  const [selectedCollection, setSelectedCollection] = useState(collection);
  const [selectedKeys, setSelectedKeys] = useState<string[]>(normalizedInitialKeys);
  const [workspaceScope, setWorkspaceScope] = useState("");
  const documents = useDocuments(workspaceScope);
  const [documentTitle, setDocumentTitle] = useState("");
  const [parserName, setParserName] = useState("");
  const [search, setSearch] = useState("");

  useEffect(() => {
    setSelectedCollection(collection);
    setSelectedKeys(normalizedInitialKeys);
  }, [collection, normalizedInitialKeys]);

  useEffect(() => {
    if (!selectedCollection && collections.data?.length) {
      const preferred =
        collections.data.find((item) => item.role === "documents") ??
        collections.data.find((item) => item.vector_type === "hybrid") ??
        collections.data[0];
      if (preferred) updateUrl(preferred.name, selectedKeys, { replaceState: true });
    }
  }, [collections.data, selectedCollection, selectedKeys]);

  const documentFilters = useMemo(
    () => ({
      clientId: workspaceScope || undefined,
      documentTitle: documentTitle || undefined,
      parserName: parserName || undefined,
      search: search.trim() || undefined,
      limit: 500,
    }),
    [documentTitle, parserName, search, workspaceScope],
  );

  const comparison = useQdrantDocumentCompare(selectedCollection, selectedKeys);
  const candidates = useQdrantDocuments(selectedCollection, documentFilters);
  const rows = useMemo(() => candidates.data?.documents ?? [], [candidates.data?.documents]);
  const validSelection = Boolean(
    selectedCollection &&
      selectedKeys.length >= 2 &&
      selectedKeys.length <= maxSelectedDocuments,
  );
  const documentSelectOptions = useMemo(
    () =>
      mergeOptions([
        ...(documents.data ?? []).map((document) => ({
          value: document.filename,
          label: document.filename,
        })),
        ...(candidates.data?.facets.documents ?? []).map((document) => ({
          value: document.title,
          label: `${document.title} (${document.count})`,
        })),
      ]),
    [candidates.data?.facets.documents, documents.data],
  );
  const parserSelectOptions = useMemo(
    () =>
      mergeOptions([
        ...parserOptions(parsers.data?.parsers),
        ...facetOptions(candidates.data?.facets.parsers, parserName),
      ]),
    [candidates.data?.facets.parsers, parserName, parsers.data?.parsers],
  );

  function updateUrl(
    nextCollection: string,
    nextKeys: string[],
    options: { replaceState?: boolean } = {},
  ) {
    const keys = normalizeDocumentKeys(nextKeys);
    setSelectedCollection(nextCollection);
    setSelectedKeys(keys);

    const params = new URLSearchParams();
    if (nextCollection) params.set("collection", nextCollection);
    for (const documentKey of keys) params.append("document_keys", documentKey);
    const query = params.toString();
    const href = query
      ? `/inspector/compare/documents?${query}`
      : "/inspector/compare/documents";
    router.replace(href, { scroll: false });
  }

  function toggleDocument(documentKey: string) {
    if (!selectedCollection) return;
    const nextKeys = selectedKeys.includes(documentKey)
      ? selectedKeys.filter((key) => key !== documentKey)
      : selectedKeys.length >= maxSelectedDocuments
        ? selectedKeys
        : [...selectedKeys, documentKey];
    if (nextKeys === selectedKeys) return;
    updateUrl(selectedCollection, nextKeys);
  }

  function changeCollection(nextCollection: string) {
    setDocumentTitle("");
    setParserName("");
    setSearch("");
    updateUrl(nextCollection, []);
  }

  function clearFilters() {
    setWorkspaceScope("");
    setDocumentTitle("");
    setParserName("");
    setSearch("");
  }

  function exportDocuments() {
    if (!comparison.data?.length) return;
    downloadJson(comparison.data, `${selectedCollection || "qdrant"}-document-compare.json`);
  }

  return (
    <div>
      <PageHeader
        title="Compare parsed documents."
        description="Review reconstructed markdown for whole indexed documents side by side, then add or remove documents below."
        actions={
          <>
            <Link className="button-secondary" href="/inspector">
              Back to inspector
            </Link>
            <Link className="button-secondary" href="/inspector/compare">
              Node compare
            </Link>
            <button
              className="button-secondary"
              disabled={!validSelection}
              onClick={() => comparison.refetch()}
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
            <button
              className="button-secondary"
              disabled={!comparison.data?.length}
              onClick={exportDocuments}
            >
              <Download className="h-4 w-4" />
              Export
            </button>
          </>
        }
      />

      <section className="mb-4">
        {!validSelection ? (
          <SectionCard>
            <div className="text-sm text-muted-foreground">
              Select 2 to 4 documents below to render the comparison.
            </div>
          </SectionCard>
        ) : comparison.isLoading ? (
          <SectionCard>
            <LoadingState />
          </SectionCard>
        ) : comparison.error ? (
          <SectionCard>
            <ErrorState error={comparison.error} title="Comparison failed" />
          </SectionCard>
        ) : comparison.data?.length ? (
          <DocumentComparison documents={comparison.data} />
        ) : (
          <SectionCard>
            <EmptyState title="No documents loaded" description="Refresh or choose documents below." />
          </SectionCard>
        )}
      </section>

      <SectionCard className="mb-4" title="Find documents">
        <div className="grid gap-3 lg:grid-cols-4 xl:grid-cols-6">
          <DarkSelect
            label="Workspace"
            value={workspaceScope}
            placeholder="Workspace"
            onChange={setWorkspaceScope}
            buttonClassName="font-mono"
            options={[
              { value: "", label: "All workspaces" },
              ...(clients.data ?? []).map((client) => ({ value: client.id, label: client.name })),
            ]}
          />
          <DarkSelect
            label="Collection"
            value={selectedCollection}
            placeholder="Collection"
            onChange={changeCollection}
            buttonClassName="font-mono"
            options={[
              { value: "", label: "Collection" },
              ...(collections.data ?? []).map((item) => ({ value: item.name, label: item.name })),
            ]}
          />
          <DarkSelect
            label="Document"
            value={documentTitle}
            placeholder="Document"
            onChange={setDocumentTitle}
            options={[
              { value: "", label: "All documents" },
              ...documentSelectOptions,
            ]}
          />
          <DarkSelect
            label="Parser"
            value={parserName}
            placeholder="Parser"
            onChange={setParserName}
            options={[
              { value: "", label: "All parsers" },
              ...parserSelectOptions,
            ]}
          />
          <div className="relative lg:col-span-2">
            <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
            <input
              className="control w-full pl-9"
              placeholder="Search document text, parser, client"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
          <button className="button-secondary" onClick={clearFilters}>
            Clear filters
          </button>
        </div>
      </SectionCard>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <SectionCard
          title={`Available documents // ${candidates.data?.total ?? 0}`}
          description="Add or remove documents from the current comparison."
          actions={
            <StatusBadge
              status={selectedKeys.length >= 2 ? "ok" : "queued"}
              label={selectedDocumentCountLabel(selectedKeys.length)}
            />
          }
        >
          {collections.error ? (
            <ErrorState error={collections.error} />
          ) : candidates.isLoading ? (
            <LoadingState />
          ) : candidates.error ? (
            <ErrorState error={candidates.error} />
          ) : rows.length ? (
            <DocumentSelectionTable
              rows={rows}
              selectedKeys={selectedKeys}
              onToggle={toggleDocument}
            />
          ) : (
            <EmptyState title="No parsed documents" description="Adjust filters or choose a populated collection." />
          )}
        </SectionCard>

        <SectionCard title="Selected documents" description="URL params preserve this order.">
          {selectedKeys.length ? (
            <div className="space-y-2">
              {selectedKeys.map((key, index) => {
                const row =
                  rows.find((document) => document.document_key === key) ??
                  comparison.data?.find((document) => document.document_key === key);
                return (
                  <div key={key} className="rounded-md border border-border bg-background p-3">
                    <div className="text-xs font-medium text-muted-foreground">
                      Document {index + 1}
                    </div>
                    <div className="mt-1 truncate font-mono text-xs">{key}</div>
                    <div className="mt-2 text-sm">
                      {row?.document_name ?? row?.document_id ?? "Selected document"}
                    </div>
                    <div className="mt-1 truncate font-mono text-[11px] text-muted-foreground">
                      client: {row?.client_id ?? "-"}
                    </div>
                    <div className="mt-1 truncate font-mono text-[11px] text-muted-foreground">
                      parser: {row?.parser_name ?? "-"}
                    </div>
                    <button
                      className="button-ghost mt-2 h-7 px-2 text-xs"
                      onClick={() => toggleDocument(key)}
                    >
                      Remove
                    </button>
                  </div>
                );
              })}
            </div>
          ) : (
            <EmptyState title="No documents selected" description="Pick rows from the document table to begin comparing parsed output." />
          )}
        </SectionCard>
      </div>
    </div>
  );
}

function normalizeDocumentKeys(documentKeys: string[]) {
  const seen = new Set<string>();
  const normalized = [];
  for (const documentKey of documentKeys) {
    const cleaned = documentKey.trim();
    if (!cleaned || seen.has(cleaned)) continue;
    seen.add(cleaned);
    normalized.push(cleaned);
    if (normalized.length >= maxSelectedDocuments) break;
  }
  return normalized;
}
