"use client";

import { Download, RefreshCw, Search, SplitSquareHorizontal } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { EmptyState } from "@/components/common/empty-state";
import { ErrorState } from "@/components/common/error-state";
import { LoadingState } from "@/components/common/loading-state";
import { SectionCard } from "@/components/common/section-card";
import { StatusBadge } from "@/components/common/status-badge";
import { DarkSelect } from "@/components/common/dark-select";
import {
  DocumentSelectionTable,
  maxSelectedDocuments,
  selectedDocumentCountLabel,
} from "@/components/qdrant/document-selection-table";
import { facetOptions, mergeOptions, parserOptions } from "@/components/qdrant/node-picker-options";
import { maxSelectedNodes, NodeSelectionTable, selectedCountLabel } from "@/components/qdrant/node-selection-table";
import { PageHeader } from "@/components/shell/page-header";
import { useClients } from "@/lib/hooks/use-clients";
import { useDocuments, useParsers } from "@/lib/hooks/use-documents";
import { useQdrantCollections, useQdrantDocuments, useQdrantNodes } from "@/lib/hooks/use-qdrant";
import { downloadJson } from "@/lib/utils";

type CompareLevel = "nodes" | "documents";

export default function QdrantInspectorPage() {
  const router = useRouter();
  const clients = useClients();
  const [compareLevel, setCompareLevel] = useState<CompareLevel>("nodes");
  const [workspaceScope, setWorkspaceScope] = useState("");
  const documents = useDocuments(workspaceScope);
  const parsers = useParsers();
  const collections = useQdrantCollections();
  const [collection, setCollection] = useState("");
  const [documentTitle, setDocumentTitle] = useState("");
  const [parserName, setParserName] = useState("");
  const [chunkType, setChunkType] = useState("");
  const [pageInput, setPageInput] = useState("");
  const [search, setSearch] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [selectedDocumentKeys, setSelectedDocumentKeys] = useState<string[]>([]);

  useEffect(() => {
    if (!collection && collections.data?.length) {
      const preferred =
        collections.data.find((item) => item.role === "documents") ??
        collections.data.find((item) => item.vector_type === "hybrid") ??
        collections.data[0];
      setCollection(preferred.name);
    }
  }, [collection, collections.data]);

  const pageNum = useMemo(() => {
    const value = Number(pageInput);
    return pageInput.trim() && Number.isFinite(value) ? value : undefined;
  }, [pageInput]);

  const nodeFilters = useMemo(
    () => ({
      clientId: workspaceScope || undefined,
      documentTitle: documentTitle || undefined,
      parserName: parserName || undefined,
      chunkType: chunkType || undefined,
      pageNum,
      search: search.trim() || undefined,
      limit: 500,
    }),
    [chunkType, documentTitle, pageNum, parserName, search, workspaceScope],
  );

  const nodes = useQdrantNodes(collection, nodeFilters);
  const rows = useMemo(() => nodes.data?.nodes ?? [], [nodes.data?.nodes]);
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
  const documentCandidates = useQdrantDocuments(collection, documentFilters);
  const documentRows = useMemo(
    () => documentCandidates.data?.documents ?? [],
    [documentCandidates.data?.documents],
  );
  const documentSelectOptions = useMemo(
    () =>
      mergeOptions([
        ...(documents.data ?? []).map((document) => ({
          value: document.filename,
          label: document.filename,
        })),
        ...((compareLevel === "documents"
          ? documentCandidates.data?.facets.documents
          : nodes.data?.facets.documents) ?? []).map((document) => ({
          value: document.title,
          label: `${document.title} (${document.count})`,
        })),
      ]),
    [compareLevel, documentCandidates.data?.facets.documents, documents.data, nodes.data?.facets.documents],
  );
  const parserSelectOptions = useMemo(
    () =>
      mergeOptions([
        ...parserOptions(parsers.data?.parsers),
        ...facetOptions(
          compareLevel === "documents"
            ? documentCandidates.data?.facets.parsers
            : nodes.data?.facets.parsers,
          parserName,
        ),
      ]),
    [compareLevel, documentCandidates.data?.facets.parsers, nodes.data?.facets.parsers, parserName, parsers.data?.parsers],
  );

  useEffect(() => {
    if (compareLevel !== "nodes") return;
    if (!rows.length) {
      setSelectedIds((current) => (current.length ? [] : current));
      return;
    }
    const visibleIds = new Set(rows.map((node) => node.id));
    setSelectedIds((current) => {
      const next = current.filter((id) => visibleIds.has(id));
      return next.length === current.length ? current : next;
    });
  }, [compareLevel, rows]);

  useEffect(() => {
    if (compareLevel !== "documents") return;
    if (!documentRows.length) {
      setSelectedDocumentKeys((current) => (current.length ? [] : current));
      return;
    }
    const visibleKeys = new Set(documentRows.map((document) => document.document_key));
    setSelectedDocumentKeys((current) => {
      const next = current.filter((key) => visibleKeys.has(key));
      return next.length === current.length ? current : next;
    });
  }, [compareLevel, documentRows]);

  function toggleNode(nodeId: string) {
    setSelectedIds((current) => {
      if (current.includes(nodeId)) return current.filter((id) => id !== nodeId);
      if (current.length >= maxSelectedNodes) return current;
      return [...current, nodeId];
    });
  }

  function toggleDocument(documentKey: string) {
    setSelectedDocumentKeys((current) => {
      if (current.includes(documentKey)) return current.filter((key) => key !== documentKey);
      if (current.length >= maxSelectedDocuments) return current;
      return [...current, documentKey];
    });
  }

  function compareSelected() {
    if (!collection) {
      return;
    }
    const params = new URLSearchParams();
    params.set("collection", collection);
    if (compareLevel === "documents") {
      if (selectedDocumentKeys.length < 2 || selectedDocumentKeys.length > maxSelectedDocuments) {
        return;
      }
      for (const documentKey of selectedDocumentKeys) params.append("document_keys", documentKey);
      router.push(`/inspector/compare/documents?${params.toString()}`);
      return;
    }
    if (selectedIds.length < 2 || selectedIds.length > maxSelectedNodes) {
      return;
    }
    for (const pointId of selectedIds) params.append("point_ids", pointId);
    router.push(`/inspector/compare?${params.toString()}`);
  }

  function clearFilters() {
    setDocumentTitle("");
    setParserName("");
    setChunkType("");
    setPageInput("");
    setSearch("");
    setSelectedIds([]);
    setSelectedDocumentKeys([]);
  }

  function exportSelected() {
    const payload =
      compareLevel === "documents"
        ? documentRows.filter((document) => selectedDocumentKeys.includes(document.document_key))
        : rows.filter((node) => selectedIds.includes(node.id));
    downloadJson(payload, `${collection || "qdrant"}-${compareLevel}-compare.json`);
  }

  const activeSelectedCount = compareLevel === "documents" ? selectedDocumentKeys.length : selectedIds.length;
  const canCompare =
    compareLevel === "documents"
      ? Boolean(collection && selectedDocumentKeys.length >= 2 && selectedDocumentKeys.length <= maxSelectedDocuments)
      : Boolean(collection && selectedIds.length >= 2 && selectedIds.length <= maxSelectedNodes);

  return (
    <div>
      <PageHeader
        title="Parsed comparison."
        description="Filter parsed index content by workspace, document, and parser, then compare nodes or reconstructed documents side by side."
        actions={
          <>
            <button
              className="button-secondary"
              onClick={() => {
                collections.refetch();
                documents.refetch();
                nodes.refetch();
                documentCandidates.refetch();
              }}
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
            <button
              className="button-primary"
              disabled={!canCompare}
              onClick={compareSelected}
            >
              <SplitSquareHorizontal className="h-4 w-4" />
              Compare
            </button>
            <button className="button-secondary" disabled={!activeSelectedCount} onClick={exportSelected}>
              <Download className="h-4 w-4" />
              Export
            </button>
          </>
        }
      />

      <SectionCard className="mb-4" title="Filters">
        <div className="grid gap-3 lg:grid-cols-4 xl:grid-cols-6">
          <div className="inline-flex h-9 rounded-full border border-border bg-background p-1">
            {(["nodes", "documents"] as const).map((level) => (
              <button
                key={level}
                className={
                  compareLevel === level
                    ? "rounded-full bg-primary px-3 text-xs font-medium text-primary-foreground"
                    : "rounded-full px-3 text-xs text-muted-foreground hover:text-foreground"
                }
                onClick={() => {
                  setCompareLevel(level);
                  setSelectedIds([]);
                  setSelectedDocumentKeys([]);
                }}
                type="button"
              >
                {level === "nodes" ? "Nodes" : "Documents"}
              </button>
            ))}
          </div>
          <DarkSelect
            label="Workspace"
            value={workspaceScope}
            placeholder="Workspace"
            onChange={(value) => {
              setWorkspaceScope(value);
              setSelectedIds([]);
              setSelectedDocumentKeys([]);
            }}
            buttonClassName="font-mono"
            options={[
              { value: "", label: "All workspaces" },
              ...(clients.data ?? []).map((client) => ({ value: client.id, label: client.name })),
            ]}
          />
          <DarkSelect
            label="Collection"
            value={collection}
            placeholder="Collection"
            onChange={(value) => {
              setCollection(value);
              setSelectedIds([]);
              setSelectedDocumentKeys([]);
            }}
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
            onChange={(value) => {
              setDocumentTitle(value);
              setSelectedIds([]);
              setSelectedDocumentKeys([]);
            }}
            options={[
              { value: "", label: "All documents" },
              ...documentSelectOptions,
            ]}
          />
          <DarkSelect
            label="Parser"
            value={parserName}
            placeholder="Parser"
            onChange={(value) => {
              setParserName(value);
              setSelectedIds([]);
              setSelectedDocumentKeys([]);
            }}
            options={[
              { value: "", label: "All parsers" },
              ...parserSelectOptions,
            ]}
          />
          {compareLevel === "nodes" ? (
            <>
              <DarkSelect
                label="Chunk type"
                value={chunkType}
                placeholder="Chunk type"
                onChange={(value) => {
                  setChunkType(value);
                  setSelectedIds([]);
                }}
                options={[
                  { value: "", label: "All chunks" },
                  ...facetOptions(nodes.data?.facets.chunk_types, chunkType),
                ]}
              />
              <input
                className="control"
                inputMode="numeric"
                placeholder="Page"
                value={pageInput}
                onChange={(event) => {
                  setPageInput(event.target.value);
                  setSelectedIds([]);
                }}
              />
            </>
          ) : null}
          <div className="relative lg:col-span-2 xl:col-span-3">
            <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
            <input
              className="control w-full pl-9"
              placeholder={
                compareLevel === "documents"
                  ? "Search document text, parser, client"
                  : "Search node text, citation, chunk, section"
              }
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setSelectedIds([]);
                setSelectedDocumentKeys([]);
              }}
            />
          </div>
          <button className="button-secondary" onClick={clearFilters}>
            Clear filters
          </button>
        </div>
      </SectionCard>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <SectionCard
          title={
            compareLevel === "documents"
              ? `Indexed documents // ${documentCandidates.data?.total ?? 0}`
              : `Indexed nodes // ${nodes.data?.total ?? 0}`
          }
          description={
            compareLevel === "documents"
              ? "Select 2 to 4 reconstructed documents for side-by-side comparison."
              : "Select 2 to 4 nodes for side-by-side comparison."
          }
          actions={
            <StatusBadge
              status={activeSelectedCount >= 2 ? "ok" : "queued"}
              label={
                compareLevel === "documents"
                  ? selectedDocumentCountLabel(selectedDocumentKeys.length)
                  : selectedCountLabel(selectedIds.length)
              }
            />
          }
        >
          {collections.error ? (
            <ErrorState error={collections.error} />
          ) : compareLevel === "documents" ? (
            documentCandidates.isLoading ? (
              <LoadingState />
            ) : documentCandidates.error ? (
              <ErrorState error={documentCandidates.error} />
            ) : documentRows.length ? (
              <DocumentSelectionTable
                rows={documentRows}
                selectedKeys={selectedDocumentKeys}
                onToggle={toggleDocument}
              />
            ) : (
              <EmptyState title="No parsed documents" description="Adjust filters or ingest documents into the selected collection." />
            )
          ) : nodes.isLoading ? (
            <LoadingState />
          ) : nodes.error ? (
            <ErrorState error={nodes.error} />
          ) : rows.length ? (
            <NodeSelectionTable rows={rows} selectedIds={selectedIds} onToggle={toggleNode} />
          ) : (
            <EmptyState title="No parsed nodes" description="Adjust filters or ingest documents into the selected collection." />
          )}
        </SectionCard>

        <SectionCard
          title="Selection"
          description={
            compareLevel === "documents"
              ? "The document compare page preserves this order."
              : "The compare panel preserves this order."
          }
        >
          {compareLevel === "documents" ? (
            selectedDocumentKeys.length ? (
              <div className="space-y-2">
                {selectedDocumentKeys.map((key, index) => {
                  const row = documentRows.find((document) => document.document_key === key);
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
                      <button className="button-ghost mt-2 h-7 px-2 text-xs" onClick={() => toggleDocument(key)}>
                        Remove
                      </button>
                    </div>
                  );
                })}
              </div>
            ) : (
              <EmptyState title="No documents selected" description="Pick rows from the document table to compare reconstructed markdown." />
            )
          ) : selectedIds.length ? (
            <div className="space-y-2">
              {selectedIds.map((id, index) => {
                const row = rows.find((node) => node.id === id);
                return (
                  <div key={id} className="rounded-md border border-border bg-background p-3">
                    <div className="text-xs font-medium text-muted-foreground">Node {index + 1}</div>
                    <div className="mt-1 truncate font-mono text-xs">{id}</div>
                    <div className="mt-2 text-sm">{row?.document_name ?? "Selected node"}</div>
                    <div className="mt-1 truncate font-mono text-[11px] text-muted-foreground">
                      client: {row?.client_id ?? "-"}
                    </div>
                    <button className="button-ghost mt-2 h-7 px-2 text-xs" onClick={() => toggleNode(id)}>
                      Remove
                    </button>
                  </div>
                );
              })}
            </div>
          ) : (
            <EmptyState title="No nodes selected" description="Pick rows from the node table to begin comparing parsed output." />
          )}
        </SectionCard>
      </div>
    </div>
  );
}
