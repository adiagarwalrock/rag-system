"""
Vector store integration with Qdrant via LlamaIndex.
Handles indexing and query engine creation.
"""

import logging
from typing import List

import qdrant_client
from google.genai import types as genai_types
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import MetadataFilters
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http import models as qdrant_models

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
    ("document_version_group", qdrant_models.PayloadSchemaType.KEYWORD),
    ("page_num", qdrant_models.PayloadSchemaType.INTEGER),
    ("slide_num", qdrant_models.PayloadSchemaType.INTEGER),
)

_llama_configured = False
_configured_key = ""
# Kept for compatibility with UI imports.
_is_placeholder = True


def _build_embedding_config() -> genai_types.EmbedContentConfig | None:
    if settings.EMBEDDING_OUTPUT_DIMENSION is None:
        return None
    return genai_types.EmbedContentConfig(
        output_dimensionality=settings.EMBEDDING_OUTPUT_DIMENSION
    )


def _configure_llama_settings() -> None:
    global _llama_configured, _configured_key, _is_placeholder

    api_key = settings.google_api_key
    if _llama_configured and api_key == _configured_key:
        return

    _configured_key = api_key
    _llama_configured = True
    _is_placeholder = settings.is_google_api_key_placeholder

    if _is_placeholder:
        Settings.llm = MockLLM()
        Settings.embed_model = MockEmbedding(embed_dim=VECTOR_DIMENSIONS)
        logger.warning(
            "GOOGLE_API_KEY missing or placeholder. Using Mock LLM/Embeddings."
        )
    else:
        Settings.llm = GoogleGenAI(model=settings.LLM_MODEL, api_key=api_key)
        Settings.embed_model = GoogleGenAIEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            api_key=api_key,
            embedding_config=_build_embedding_config(),
        )
        logger.info(
            "GOOGLE_API_KEY configured (prefix=%s). Using Google GenAI models "
            "(llm=%s, embedding=%s).",
            api_key[:7],
            settings.LLM_MODEL,
            settings.EMBEDDING_MODEL,
        )


def is_placeholder_mode() -> bool:
    _configure_llama_settings()
    return _is_placeholder


# Lazy singletons
_client = None
_vector_store = None
_storage_context = None
_indexes_ensured = False


def _get_qdrant_client():
    global _client
    if _client is None:
        _configure_llama_settings()
        _client = qdrant_client.QdrantClient(
            url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
        )
        logger.info("Connected to Qdrant at %s", settings.QDRANT_URL)
    return _client


def get_qdrant_client():
    """Return the configured Qdrant client."""
    return _get_qdrant_client()


def reset_vector_store_cache() -> None:
    """Reset cached Qdrant/LlamaIndex objects after collection recreation."""
    global _vector_store, _storage_context, _indexes_ensured
    _vector_store = None
    _storage_context = None
    _indexes_ensured = False


def collection_has_sparse_vectors(
    client=None, collection_name: str = COLLECTION_NAME
) -> bool:
    """Return whether a Qdrant collection has the configured sparse vector slot."""
    qdrant = client or _get_qdrant_client()
    if not qdrant.collection_exists(collection_name):
        return False
    info = qdrant.get_collection(collection_name)
    sparse_vectors = info.config.params.sparse_vectors or {}
    return SPARSE_VECTOR_NAME in sparse_vectors


def create_hybrid_collection(
    client=None, collection_name: str = COLLECTION_NAME
) -> None:
    """Create the expected dense+sparse Qdrant collection."""
    qdrant = client or _get_qdrant_client()
    qdrant.create_collection(
        collection_name=collection_name,
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


def _ensure_collection_exists(client):
    if not client.collection_exists(COLLECTION_NAME):
        create_hybrid_collection(client, COLLECTION_NAME)
        logger.info(
            "Created hybrid collection '%s' (%d dim dense + sparse BM25)",
            COLLECTION_NAME,
            VECTOR_DIMENSIONS,
        )
    else:
        _ensure_vector_dimensions_match(client)
        _ensure_sparse_vectors(client)

    _ensure_payload_indexes(client)


def _ensure_vector_dimensions_match(client) -> None:
    """Fail fast when existing collection dimensions do not match current config."""
    info = client.get_collection(COLLECTION_NAME)
    vectors_config = info.config.params.vectors

    vector_params = None
    if isinstance(vectors_config, dict):
        vector_params = vectors_config.get(DENSE_VECTOR_NAME)
    elif hasattr(vectors_config, "size"):
        vector_params = vectors_config

    actual_dim = getattr(vector_params, "size", None)
    if actual_dim is None:
        logger.warning(
            "Could not determine dense vector dimensions for collection '%s'.",
            COLLECTION_NAME,
        )
        return

    if actual_dim != VECTOR_DIMENSIONS:
        raise RuntimeError(
            "Collection '%s' has dense vector size %s but config expects %s. "
            "Recreate the collection (development) or align VECTOR_DIMENSIONS/"
            "EMBEDDING_OUTPUT_DIMENSION."
            % (COLLECTION_NAME, actual_dim, VECTOR_DIMENSIONS)
        )


def _ensure_sparse_vectors(client) -> None:
    """Warn when an existing collection cannot support hybrid sparse search."""
    if collection_has_sparse_vectors(client, COLLECTION_NAME):
        return

    logger.warning(
        "Collection '%s' does not have sparse vector slot '%s'. "
        "Qdrant cannot add this vector name in-place for this collection. "
        "Recreate the collection to restore hybrid behavior, or continue with "
        "dense fallback.",
        COLLECTION_NAME,
        SPARSE_VECTOR_NAME,
    )


def _collection_supports_hybrid(client) -> bool:
    try:
        return collection_has_sparse_vectors(client, COLLECTION_NAME)
    except Exception:
        logger.exception(
            "Could not inspect sparse vector config for collection '%s'",
            COLLECTION_NAME,
        )
        return False


def _ensure_payload_indexes(client) -> None:
    global _indexes_ensured
    if _indexes_ensured:
        return

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
            return  # Stop if one fails, but don't set _indexes_ensured to True

    _indexes_ensured = True


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        client = _get_qdrant_client()
        _ensure_collection_exists(client)
        hybrid_enabled = _collection_supports_hybrid(client)
        _vector_store = QdrantVectorStore(
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
    return _vector_store


def _get_storage_context():
    global _storage_context
    if _storage_context is None:
        _storage_context = StorageContext.from_defaults(
            vector_store=_get_vector_store()
        )
    return _storage_context


def get_index() -> VectorStoreIndex:
    """Return a VectorStoreIndex backed by the configured Qdrant store."""
    return VectorStoreIndex.from_vector_store(vector_store=_get_vector_store())


def get_retriever(
    filters: MetadataFilters | None = None,
    similarity_top_k: int = 15,
    sparse_top_k: int | None = None,
    hybrid_top_k: int | None = None,
    hybrid: bool = True,
):
    """Get a retriever with optional hybrid dense+sparse Qdrant search."""
    index = get_index()
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


def index_nodes(nodes: List[BaseNode]) -> None:
    """Persist transformed nodes to Qdrant."""
    if not nodes:
        return

    _ensure_collection_exists(_get_qdrant_client())

    for node in nodes:
        # Clean up redundant filename metadata
        if node.metadata.get("file_name") == node.metadata.get("filename"):
            node.metadata.pop("filename", None)

    VectorStoreIndex(nodes=nodes, storage_context=_get_storage_context())
    logger.info("Indexed %d nodes into '%s'", len(nodes), COLLECTION_NAME)


def safe_upsert_vector_store(nodes: List[BaseNode]) -> bool:
    """Persist transformed nodes to Qdrant safely, bubbling up exceptions."""
    try:
        index_nodes(nodes)
        return True
    except Exception as e:
        logger.exception("Failed to safely upsert nodes to vector store")
        raise e


def delete_document_vectors(document_id: str, client_id: str) -> bool:
    """Delete all vectors for a specific document from Qdrant by payload filter."""
    try:
        client = _get_qdrant_client()
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
    except Exception as e:
        logger.exception(
            "Failed to delete vectors for document_id='%s': %s", document_id, e
        )
        return False


def get_qdrant_point_count() -> int | None:
    """Return approximate point count for the configured collection."""
    try:
        client = _get_qdrant_client()
        _ensure_collection_exists(client)
        return client.count(collection_name=COLLECTION_NAME).count
    except Exception:
        logger.exception("Failed to read Qdrant point count")
        return None
