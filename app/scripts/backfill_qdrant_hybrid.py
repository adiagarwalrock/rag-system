"""
Backfill existing Qdrant points with sparse vectors for hybrid search.

This reads nodes from Qdrant payload `_node_content` and re-upserts them through
the hybrid-enabled vector store. It does not read chunk text from SQL/Snowflake.
"""

import argparse
import logging

from llama_index.core.vector_stores.utils import metadata_dict_to_node
from qdrant_client.http import models as qdrant_models

from app.indexing.vector_store import (
    COLLECTION_NAME,
    collection_has_sparse_vectors,
    create_hybrid_collection,
    get_qdrant_client,
    index_nodes,
    reset_vector_store_cache,
)

logger = logging.getLogger(__name__)


def backfill_hybrid_vectors(
    client_id: str | None = None,
    document_id: str | None = None,
    batch_size: int = 64,
    dry_run: bool = False,
    recreate_collection: bool = False,
) -> int:
    qdrant = get_qdrant_client()
    if not qdrant.collection_exists(COLLECTION_NAME):
        logger.info(
            "Collection '%s' does not exist; nothing to backfill.", COLLECTION_NAME
        )
        return 0

    supports_sparse = collection_has_sparse_vectors(qdrant, COLLECTION_NAME)
    if not supports_sparse and not recreate_collection:
        message = (
            f"Collection '{COLLECTION_NAME}' is dense-only and cannot accept sparse "
            "vectors in-place. Re-run with --recreate-collection to rebuild the "
            "collection from Qdrant payloads, or reingest documents into a fresh "
            "hybrid collection."
        )
        if dry_run:
            logger.warning(message)
        else:
            raise RuntimeError(message)

    # Recreating the collection affects every point in the collection, so preserve
    # all recoverable nodes regardless of scoped filters.
    filter_for_scroll = (
        None if recreate_collection else _build_filter(client_id, document_id)
    )
    nodes = _load_nodes_from_qdrant(qdrant, filter_for_scroll, batch_size)

    if dry_run:
        return len(nodes)

    if recreate_collection:
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
        create_hybrid_collection(qdrant, COLLECTION_NAME)
        reset_vector_store_cache()

    if nodes:
        index_nodes(nodes)
    return len(nodes)


def _load_nodes_from_qdrant(qdrant, scroll_filter, batch_size: int) -> list:
    offset = None
    nodes = []
    while True:
        records, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break

        for record in records:
            payload = record.payload or {}
            try:
                node = metadata_dict_to_node(payload)
            except Exception:
                logger.warning(
                    "Skipping point %s without recoverable node payload", record.id
                )
                continue
            nodes.append(node)

        if offset is None:
            break

    return nodes


def _build_filter(client_id: str | None, document_id: str | None):
    must = []
    if client_id:
        must.append(
            qdrant_models.FieldCondition(
                key="client_id",
                match=qdrant_models.MatchValue(value=client_id),
            )
        )
    if document_id:
        must.append(
            qdrant_models.FieldCondition(
                key="document_id",
                match=qdrant_models.MatchValue(value=document_id),
            )
        )
    if not must:
        return None
    return qdrant_models.Filter(must=must)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill Qdrant sparse vectors for hybrid retrieval."
    )
    parser.add_argument("--client-id")
    parser.add_argument("--document-id")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help=(
            "Delete and recreate the Qdrant collection with dense+sparse schema, "
            "then re-upsert recoverable nodes from Qdrant payloads. This preserves "
            "all recoverable points, not only scoped --client-id/--document-id points."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        count = backfill_hybrid_vectors(
            client_id=args.client_id,
            document_id=args.document_id,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            recreate_collection=args.recreate_collection,
        )
    except RuntimeError as exc:
        parser.exit(status=2, message=f"ERROR: {exc}\n")
    action = "Would backfill" if args.dry_run else "Backfilled"
    logger.info("%s %d Qdrant point(s)", action, count)


if __name__ == "__main__":
    main()
