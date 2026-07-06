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
import { NodeComparison } from "@/components/qdrant/node-comparison";
import { facetOptions, mergeOptions, parserOptions } from "@/components/qdrant/node-picker-options";
import { maxSelectedNodes, NodeSelectionTable, selectedCountLabel } from "@/components/qdrant/node-selection-table";
import { PageHeader } from "@/components/shell/page-header";
import { useClients } from "@/lib/hooks/use-clients";
import { useDocuments, useParsers } from "@/lib/hooks/use-documents";
import { useQdrantCollections, useQdrantNodeCompare, useQdrantNodes } from "@/lib/hooks/use-qdrant";
import { downloadJson } from "@/lib/utils";

export function NodeCompareRoute({
  collection,
  pointIds,
}: {
  collection: string;
  pointIds: string[];
}) {
  const router = useRouter();
  const normalizedInitialIds = useMemo(() => normalizePointIds(pointIds), [pointIds]);
  const clients = useClients();
  const parsers = useParsers();
  const collections = useQdrantCollections();
  const [selectedCollection, setSelectedCollection] = useState(collection);
  const [selectedIds, setSelectedIds] = useState<string[]>(normalizedInitialIds);
  const [workspaceScope, setWorkspaceScope] = useState("");
  const documents = useDocuments(workspaceScope);
  const [documentTitle, setDocumentTitle] = useState("");
  const [parserName, setParserName] = useState("");
  const [chunkType, setChunkType] = useState("");
  const [pageInput, setPageInput] = useState("");
  const [search, setSearch] = useState("");

  useEffect(() => {
    setSelectedCollection(collection);
    setSelectedIds(normalizedInitialIds);
  }, [collection, normalizedInitialIds]);

  useEffect(() => {
    if (!selectedCollection && collections.data?.length) {
      const preferred =
        collections.data.find((item) => item.role === "documents") ??
        collections.data.find((item) => item.vector_type === "hybrid") ??
        collections.data[0];
      if (preferred) updateUrl(preferred.name, selectedIds, { replaceState: true });
    }
  }, [collections.data, selectedCollection, selectedIds]);

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

  const comparison = useQdrantNodeCompare(selectedCollection, selectedIds);
  const nodes = useQdrantNodes(selectedCollection, nodeFilters);
  const rows = useMemo(() => nodes.data?.nodes ?? [], [nodes.data?.nodes]);
  const validSelection = Boolean(
    selectedCollection && selectedIds.length >= 2 && selectedIds.length <= maxSelectedNodes,
  );
  const documentSelectOptions = useMemo(
    () =>
      mergeOptions([
        ...(documents.data ?? []).map((document) => ({
          value: document.filename,
          label: document.filename,
        })),
        ...(nodes.data?.facets.documents ?? []).map((document) => ({
          value: document.title,
          label: `${document.title} (${document.count})`,
        })),
      ]),
    [documents.data, nodes.data?.facets.documents],
  );
  const parserSelectOptions = useMemo(
    () =>
      mergeOptions([
        ...parserOptions(parsers.data?.parsers),
        ...facetOptions(nodes.data?.facets.parsers, parserName),
      ]),
    [nodes.data?.facets.parsers, parserName, parsers.data?.parsers],
  );

  function updateUrl(
    nextCollection: string,
    nextIds: string[],
    options: { replaceState?: boolean } = {},
  ) {
    const ids = normalizePointIds(nextIds);
    setSelectedCollection(nextCollection);
    setSelectedIds(ids);

    const params = new URLSearchParams();
    if (nextCollection) params.set("collection", nextCollection);
    for (const pointId of ids) params.append("point_ids", pointId);
    const query = params.toString();
    const href = query ? `/inspector/compare?${query}` : "/inspector/compare";
    router.replace(href, { scroll: false });
  }

  function toggleNode(nodeId: string) {
    if (!selectedCollection) return;
    const nextIds = selectedIds.includes(nodeId)
      ? selectedIds.filter((id) => id !== nodeId)
      : selectedIds.length >= maxSelectedNodes
        ? selectedIds
        : [...selectedIds, nodeId];
    if (nextIds === selectedIds) return;
    updateUrl(selectedCollection, nextIds);
  }

  function changeCollection(nextCollection: string) {
    setDocumentTitle("");
    setParserName("");
    setChunkType("");
    setPageInput("");
    setSearch("");
    updateUrl(nextCollection, []);
  }

  function clearFilters() {
    setWorkspaceScope("");
    setDocumentTitle("");
    setParserName("");
    setChunkType("");
    setPageInput("");
    setSearch("");
  }

  function exportNodes() {
    if (!comparison.data?.length) return;
    downloadJson(comparison.data, `${selectedCollection || "qdrant"}-node-compare.json`);
  }

  return (
    <div>
      <PageHeader
        title="Compare parsed nodes."
        description="Review selected node text and metadata side by side, then add or remove nodes below."
        actions={
          <>
            <Link className="button-secondary" href="/inspector">
              Back to inspector
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
              onClick={exportNodes}
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
              Select 2 to 4 nodes below to render the comparison.
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
          <NodeComparison nodes={comparison.data} />
        ) : (
          <SectionCard>
            <EmptyState title="No nodes loaded" description="Refresh or choose nodes below." />
          </SectionCard>
        )}
      </section>

      <SectionCard className="mb-4" title="Find nodes">
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
          <DarkSelect
            label="Chunk type"
            value={chunkType}
            placeholder="Chunk type"
            onChange={setChunkType}
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
            onChange={(event) => setPageInput(event.target.value)}
          />
          <div className="relative lg:col-span-2 xl:col-span-3">
            <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
            <input
              className="control w-full pl-9"
              placeholder="Search node text, citation, chunk, section"
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
          title={`Available nodes // ${nodes.data?.total ?? 0}`}
          description="Add or remove nodes from the current comparison."
          actions={<StatusBadge status={selectedIds.length >= 2 ? "ok" : "queued"} label={selectedCountLabel(selectedIds.length)} />}
        >
          {collections.error ? (
            <ErrorState error={collections.error} />
          ) : nodes.isLoading ? (
            <LoadingState />
          ) : nodes.error ? (
            <ErrorState error={nodes.error} />
          ) : rows.length ? (
            <NodeSelectionTable rows={rows} selectedIds={selectedIds} onToggle={toggleNode} />
          ) : (
            <EmptyState title="No parsed nodes" description="Adjust filters or choose a populated collection." />
          )}
        </SectionCard>

        <SectionCard title="Selected nodes" description="URL params preserve this order.">
          {selectedIds.length ? (
            <div className="space-y-2">
              {selectedIds.map((id, index) => {
                const row = rows.find((node) => node.id === id) ?? comparison.data?.find((node) => node.id === id);
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

function normalizePointIds(pointIds: string[]) {
  const seen = new Set<string>();
  const normalized = [];
  for (const pointId of pointIds) {
    const cleaned = pointId.trim();
    if (!cleaned || seen.has(cleaned)) continue;
    seen.add(cleaned);
    normalized.push(cleaned);
    if (normalized.length >= maxSelectedNodes) break;
  }
  return normalized;
}
