import { NodeCompareRoute } from "@/components/qdrant/node-compare-route";
import { firstParam, arrayParam } from "@/lib/utils";

export default async function InspectorComparePage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const collection = firstParam(params.collection);
  const pointIds = arrayParam(params.point_ids).filter(Boolean);

  return <NodeCompareRoute collection={collection} pointIds={pointIds} />;
}
