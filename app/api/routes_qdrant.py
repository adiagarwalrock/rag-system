from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.core.safe_coerce import safe_int
from app.indexing.vector_store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    vector_store_manager,
)

router = APIRouter()


@router.get("/collections")
def list_collections():
    client = vector_store_manager.get_qdrant_client()
    response = client.get_collections()
    collections = getattr(response, "collections", []) or []

    rows: list[dict[str, Any]] = []
    for collection in collections:
        name = str(getattr(collection, "name", collection))
        info = None
        point_count = 0
        vector_type = "dense"
        health = "ok"
        try:
            info = client.get_collection(name)
            point_count = int(getattr(info, "points_count", None) or 0)
            vector_type = _collection_vector_type(info)
        except Exception:
            health = "degraded"

        rows.append(
            {
                "name": name,
                "role": _collection_role(name),
                "point_count": point_count,
                "vector_type": vector_type,
                "health": health,
                "last_updated": None,
            }
        )

    return rows


@router.get("/collections/{collection_name}/points")
def list_points(
    collection_name: str,
    limit: int = Query(200, ge=1, le=1000),
    client_id: str | None = Query(None),
):
    client = vector_store_manager.get_qdrant_client()
    try:
        if not client.collection_exists(collection_name):
            raise HTTPException(status_code=404, detail="Collection not found")

        scroll_filter = None
        if client_id:
            from qdrant_client.http import models as qdrant_models

            scroll_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="client_id",
                        match=qdrant_models.MatchValue(value=client_id),
                    )
                ]
            )

        records, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return [_serialize_point(collection_name, record) for record in records]


@router.get("/collections/{collection_name}/points/{point_id}")
def get_point(collection_name: str, point_id: str):
    client = vector_store_manager.get_qdrant_client()
    try:
        records = client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not records:
        raise HTTPException(status_code=404, detail="Point not found")
    return _serialize_point(collection_name, records[0])


def _collection_vector_type(info: Any) -> str:
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    sparse_vectors = getattr(params, "sparse_vectors", None) or {}
    has_dense = bool(vectors)
    has_sparse = bool(sparse_vectors)
    if has_dense and has_sparse:
        return "hybrid"
    if has_sparse:
        return "sparse"
    return "dense"


def _collection_role(name: str) -> str:
    if name == settings.COLLECTION_NAME:
        return "documents"
    if name == settings.CHAT_HISTORY_COLLECTION_NAME:
        return "memory"
    return "other"


def _serialize_point(collection_name: str, record: Any) -> dict[str, Any]:
    payload = getattr(record, "payload", None) or {}
    vector = getattr(record, "vector", None)
    node_payload = _node_payload(payload)
    metadata = node_payload.get("metadata") or {}

    dense_size = _dense_vector_size(vector)
    sparse_available = _sparse_vector_available(vector)

    return {
        "id": str(getattr(record, "id", "")),
        "collection": collection_name,
        "document_id": _first_text(metadata, payload, "document_id"),
        "filename": _first_text(
            metadata,
            payload,
            "file_name",
            "filename",
            "document_name",
            "source_file",
        ),
        "page": _first_int(metadata, payload, "page_num", "page", "slide_num"),
        "chunk_preview": _chunk_preview(node_payload, payload),
        "dense_vector_size": dense_size,
        "sparse_vector_available": sparse_available,
        "document_family": _first_text(
            metadata,
            payload,
            "document_version_group",
            "document_family",
        ),
        "version": _first_text(metadata, payload, "version_label", "version"),
        "payload": payload,
        "score_breakdown": {},
    }


def _node_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("_node_content")
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _chunk_preview(node_payload: dict[str, Any], payload: dict[str, Any]) -> str | None:
    text = node_payload.get("text")
    if not isinstance(text, str):
        text_resource = node_payload.get("text_resource")
        if isinstance(text_resource, dict):
            text = text_resource.get("text")
    if not isinstance(text, str):
        text = payload.get("text") or payload.get("chunk_text")
    if not isinstance(text, str):
        return None
    return " ".join(text.split())[:500]


def _dense_vector_size(vector: Any) -> int | None:
    dense = _named_vector(vector, DENSE_VECTOR_NAME)
    if dense is None and isinstance(vector, list):
        dense = vector
    if isinstance(dense, list):
        return len(dense)
    return None


def _sparse_vector_available(vector: Any) -> bool:
    sparse = _named_vector(vector, SPARSE_VECTOR_NAME)
    if sparse is None:
        return False
    indices = getattr(sparse, "indices", None)
    if indices is not None:
        return len(indices) > 0
    if isinstance(sparse, dict):
        return bool(sparse.get("indices") or sparse.get("values"))
    return True


def _named_vector(vector: Any, name: str) -> Any:
    if isinstance(vector, dict):
        return vector.get(name)
    return getattr(vector, name, None)


def _first_text(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *keys: str,
) -> str | None:
    for source in (primary, secondary):
        for key in keys:
            value = source.get(key)
            if value not in (None, "", "None", "null"):
                return str(value)
    return None


def _first_int(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *keys: str,
) -> int | None:
    return safe_int(_first_text(primary, secondary, *keys))
