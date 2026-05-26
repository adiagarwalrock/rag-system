"use client";

import { Download, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { EmptyState } from "@/components/common/empty-state";
import { ErrorState } from "@/components/common/error-state";
import { LoadingState } from "@/components/common/loading-state";
import { SectionCard } from "@/components/common/section-card";
import { DarkSelect } from "@/components/common/dark-select";
import { JsonDrawer } from "@/components/qdrant/json-drawer";
import { QdrantPointTable } from "@/components/qdrant/qdrant-point-table";
import { PageHeader } from "@/components/shell/page-header";
import { useQdrantCollections, useQdrantPoints } from "@/lib/hooks/use-qdrant";
import type { QdrantPoint } from "@/lib/api/schemas";

export default function QdrantPage() {
  const collections = useQdrantCollections();
  const [collection, setCollection] = useState("");
  const [search, setSearch] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [vectorMode, setVectorMode] = useState<"hybrid" | "sparse" | "dense">("hybrid");
  const [selected, setSelected] = useState<QdrantPoint | null>(null);
  const points = useQdrantPoints(collection);

  useEffect(() => {
    if (!collection && collections.data?.length) {
      const preferred =
        collections.data.find((item) => item.vector_type === "hybrid") ??
        collections.data[0];
      setCollection(preferred.name);
    }
  }, [collection, collections.data]);

  const filtered = useMemo(
    () =>
      (points.data ?? []).filter((point) => {
        const hasDense = Boolean(point.dense_vector_size);
        const hasSparse = Boolean(point.sparse_vector_available);
        if (vectorMode === "hybrid" && !(hasDense && hasSparse)) return false;
        if (vectorMode === "sparse" && !hasSparse) return false;
        if (vectorMode === "dense" && !(hasDense && !hasSparse)) return false;
        if (search && !JSON.stringify(point).toLowerCase().includes(search.toLowerCase())) return false;
        if (documentId && point.document_id !== documentId) return false;
        return true;
      }),
    [points.data, search, documentId, vectorMode],
  );

  function exportJson() {
    const blob = new Blob([JSON.stringify(filtered, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${collection || "qdrant"}-points.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div>
      <PageHeader
        title="Qdrant Inspector."
        description="Inspect vector points, payload metadata, document groupings, and retrieval score details."
        actions={
          <>
            <button className="button-secondary" onClick={() => { collections.refetch(); points.refetch(); }}>
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
            <button className="button-secondary" onClick={exportJson}>
              <Download className="h-4 w-4" />
              Export JSON
            </button>
          </>
        }
      />
      <SectionCard className="mb-4">
        <div className="grid gap-3 md:grid-cols-5">
          <DarkSelect
            label="Collection"
            value={collection}
            placeholder="Collection"
            onChange={setCollection}
            buttonClassName="font-mono"
            options={[
              { value: "", label: "Collection" },
              ...(collections.data ?? []).map((item) => ({ value: item.name, label: item.name })),
            ]}
          />
          <input className="control md:col-span-2" placeholder="Search metadata" value={search} onChange={(event) => setSearch(event.target.value)} />
          <input className="control" placeholder="Document ID" value={documentId} onChange={(event) => setDocumentId(event.target.value)} />
          <DarkSelect
            label="Vector mode"
            value={vectorMode}
            onChange={setVectorMode}
            options={[
              { value: "hybrid", label: "hybrid" },
              { value: "sparse", label: "sparse" },
              { value: "dense", label: "dense" },
            ]}
          />
        </div>
      </SectionCard>
      {collections.error ? <ErrorState error={collections.error} /> : points.isLoading ? <LoadingState /> : points.error ? <ErrorState error={points.error} /> : filtered.length ? <QdrantPointTable points={filtered} onSelect={setSelected} /> : <EmptyState title="No vector points" description="Select a Qdrant collection to inspect points and payload metadata." />}
      <JsonDrawer open={Boolean(selected)} title={selected?.id ?? "Point"} value={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
