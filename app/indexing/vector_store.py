"""
Vector store integration with Qdrant via LlamaIndex.
"""

import logging
from typing import List

import qdrant_client
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import MetadataFilters
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http import models as qdrant_models

from app.core.ai_provider import initialize_ai_provider
from app.core.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.COLLECTION_NAME
VECTOR_DIMENSIONS = settings.effective_vector_dimensions
DENSE_VECTOR_NAME = "text-dense"
SPARSE_VECTOR_NAME = "text-sparse-new"
SPARSE_MODEL_NAME = "Qdrant/bm25"
PAYLOAD_INDEXES = (
    ("client_id", qdrant_models.PayloadSchemaType.KEYWORD),
    ("document_id", qdrant_models.PayloadSchemaType.KEYWORD),
    ("file_name", qdrant_models.PayloadSchemaType.KEYWORD),
    ("document_name", qdrant_models.PayloadSchemaType.KEYWORD),
    ("parser_name", qdrant_models.PayloadSchemaType.KEYWORD),
    ("chunk_type", qdrant_models.PayloadSchemaType.KEYWORD),
    ("document_version_group", qdrant_models.PayloadSchemaType.KEYWORD),
    ("document_type", qdrant_models.PayloadSchemaType.KEYWORD),
    ("page_num", qdrant_models.PayloadSchemaType.INTEGER),
    ("slide_num", qdrant_models.PayloadSchemaType.INTEGER),
)


class VectorStoreManager:
    """Singleton manager for Llama settings and Qdrant-backed vector operations."""

    _instance: "VectorStoreManager | None" = None

    def __init__(self) -> None:
        self._client = None
        self._vector_store = None
        self._storage_context = None
        self._indexes_ensured = False

    @classmethod
    def instance(cls) -> "VectorStoreManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure_llama_settings(self) -> None:
        """Backward-compatible wrapper for centralized provider initialization."""
        initialize_ai_provider()

    def _get_qdrant_client(self):
        if self._client is None:
            self._client = qdrant_client.QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
            logger.info("Connected to Qdrant at %s", settings.QDRANT_URL)
        return self._client

    def get_qdrant_client(self):
        """Return the configured Qdrant client."""
        return self._get_qdrant_client()

    def _ensure_collection_and_indexes(self, client) -> bool:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    DENSE_VECTOR_NAME: qdrant_models.VectorParams(
                        size=VECTOR_DIMENSIONS,
                        distance=qdrant_models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: qdrant_models.SparseVectorParams(
                        index=qdrant_models.SparseIndexParams(),
                        modifier=qdrant_models.Modifier.IDF,
                    )
                },
            )
            logger.info(
                "Created hybrid collection '%s' (%d dim dense + sparse BM25)",
                COLLECTION_NAME,
                VECTOR_DIMENSIONS,
            )
            hybrid_enabled = True
        else:
            info = client.get_collection(COLLECTION_NAME)
            vectors_config = info.config.params.vectors
            vector_params = None
            if isinstance(vectors_config, dict):
                vector_params = vectors_config.get(DENSE_VECTOR_NAME)
            elif hasattr(vectors_config, "size"):
                vector_params = vectors_config

            actual_dim = getattr(vector_params, "size", None)
            if actual_dim is not None and actual_dim != VECTOR_DIMENSIONS:
                raise RuntimeError(
                    "Collection '%s' has dense vector size %s but config expects %s. "
                    "Recreate the collection (development) or align VECTOR_DIMENSIONS/"
                    "EMBEDDING_OUTPUT_DIMENSION."
                    % (COLLECTION_NAME, actual_dim, VECTOR_DIMENSIONS)
                )

            sparse_vectors = info.config.params.sparse_vectors or {}
            hybrid_enabled = SPARSE_VECTOR_NAME in sparse_vectors
            if not hybrid_enabled:
                logger.warning(
                    "Collection '%s' does not have sparse vector slot '%s'. "
                    "Qdrant cannot add this vector name in-place for this collection. "
                    "Recreate the collection to restore hybrid behavior, or continue with "
                    "dense fallback.",
                    COLLECTION_NAME,
                    SPARSE_VECTOR_NAME,
                )

        if not self._indexes_ensured:
            for field_name, field_schema in PAYLOAD_INDEXES:
                try:
                    client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=field_name,
                        field_schema=field_schema,
                        wait=True,
                    )
                except Exception:
                    logger.exception("Failed to ensure index for '%s'", field_name)
                    return hybrid_enabled
            self._indexes_ensured = True

        return hybrid_enabled

    def _get_vector_store(self):
        if self._vector_store is None:
            client = self._get_qdrant_client()
            hybrid_enabled = self._ensure_collection_and_indexes(client)
            self._vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                text_key="text",
                enable_hybrid=hybrid_enabled,
                fastembed_sparse_model=SPARSE_MODEL_NAME,
                dense_vector_name=DENSE_VECTOR_NAME,
                sparse_vector_name=SPARSE_VECTOR_NAME,
            )
            if not hybrid_enabled:
                logger.warning(
                    "Using dense-only QdrantVectorStore for '%s' until the collection is rebuilt.",
                    COLLECTION_NAME,
                )
        return self._vector_store

    def get_retriever(
        self,
        filters: MetadataFilters | None = None,
        similarity_top_k: int = 15,
        sparse_top_k: int | None = None,
        hybrid_top_k: int | None = None,
        hybrid: bool = True,
        embed_model: object | None = None,
    ):
        """Get a retriever with optional hybrid dense+sparse Qdrant search."""
        index_kwargs: dict = {"vector_store": self._get_vector_store()}
        if embed_model is not None:
            index_kwargs["embed_model"] = embed_model
        index = VectorStoreIndex.from_vector_store(**index_kwargs)
        kwargs = {
            "filters": filters,
            "similarity_top_k": similarity_top_k,
        }
        if hybrid:
            kwargs.update(
                {
                    "vector_store_query_mode": "hybrid",
                    "sparse_top_k": sparse_top_k or similarity_top_k,
                    "hybrid_top_k": hybrid_top_k or similarity_top_k,
                }
            )
        return index.as_retriever(**kwargs)

    def index_nodes(self, nodes: List[BaseNode]) -> None:
        """Persist transformed nodes to Qdrant."""
        if not nodes:
            return

        client = self._get_qdrant_client()
        self._ensure_collection_and_indexes(client)

        for node in nodes:
            metadata = node.metadata or {}
            if metadata.get("file_name") == metadata.get("filename"):
                metadata.pop("filename", None)
            node.metadata = metadata

        if self._storage_context is None:
            self._storage_context = StorageContext.from_defaults(
                vector_store=self._get_vector_store()
            )

        VectorStoreIndex(nodes=nodes, storage_context=self._storage_context)
        logger.info("Indexed %d nodes into '%s'", len(nodes), COLLECTION_NAME)

    def delete_document_vectors(self, document_id: str, client_id: str) -> bool:
        """Delete all vectors for a specific document from Qdrant by payload filter."""
        try:
            client = self._get_qdrant_client()
            if not client.collection_exists(COLLECTION_NAME):
                return True

            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=document_id),
                        ),
                        qdrant_models.FieldCondition(
                            key="client_id",
                            match=qdrant_models.MatchValue(value=client_id),
                        ),
                    ]
                ),
            )
            logger.info(
                "Deleted vectors from Qdrant for document_id='%s', client_id='%s'",
                document_id,
                client_id,
            )
            return True
        except Exception as exc:
            logger.exception(
                "Failed to delete vectors for document_id='%s': %s",
                document_id,
                exc,
            )
            return False

    def get_qdrant_point_count(self) -> int | None:
        """Return approximate point count for the configured collection."""
        try:
            client = self._get_qdrant_client()
            self._ensure_collection_and_indexes(client)
            return client.count(collection_name=COLLECTION_NAME).count
        except Exception:
            logger.exception("Failed to read Qdrant point count")
            return None


vector_store_manager = VectorStoreManager.instance()
