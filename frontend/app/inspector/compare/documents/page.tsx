import { DocumentCompareRoute } from "@/components/qdrant/document-compare-route";
import { firstParam, arrayParam } from "@/lib/utils";

export default async function InspectorDocumentComparePage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const collection = firstParam(params.collection);
  const documentKeys = arrayParam(params.document_keys).filter(Boolean);

  return <DocumentCompareRoute collection={collection} documentKeys={documentKeys} />;
}
