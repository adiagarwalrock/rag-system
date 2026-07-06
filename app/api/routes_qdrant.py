from __future__ import annotations

import json
from typing import Any

from qdrant_client.http import models as qdrant_models

from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.core.safe_coerce import safe_int
from app.api.qdrant_document_compare import (
    DOCUMENT_COMPARE_MIN,
    DOCUMENT_LIST_SCAN_LIMIT,
    compare_documents as compare_qdrant_documents,
    list_document_candidates,
)
from app.indexing.vector_store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    vector_store_manager,
)

router = APIRouter()
_MISSING_SENTINELS = {"", "none", "null", "n/a", "na", "undefined"}
_NODE_TEXT_PREVIEW_CHARS = 500
_NODE_COMPARE_MIN = 2
_NODE_COMPARE_MAX = 4


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

        scroll_filter = _build_node_scroll_filter(
            client_id=client_id,
            document_id=None,
            parser_name=None,
            chunk_type=None,
            page_num=None,
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


@router.get("/collections/{collection_name}/nodes")
def list_nodes(
    collection_name: str,
    limit: int = Query(200, ge=1, le=1000),
    client_id: str | None = Query(None),
    document_id: str | None = Query(None),
    document_title: str | None = Query(None),
    parser_name: str | None = Query(None),
    chunk_type: str | None = Query(None),
    page_num: int | None = Query(None),
    search: str | None = Query(None),
):
    client = vector_store_manager.get_qdrant_client()
    try:
        if not client.collection_exists(collection_name):
            raise HTTPException(status_code=404, detail="Collection not found")

        scroll_filter = _build_node_scroll_filter(
            client_id=client_id,
            document_id=document_id,
            parser_name=parser_name,
            chunk_type=chunk_type,
            page_num=page_num,
        )
        records = _scroll_node_records(
            client,
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            post_filter_enabled=bool(document_title or search),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    normalized = [
        _serialize_node(collection_name, record, include_raw_payload=False)
        for record in records
    ]
    filtered = [
        node
        for node in normalized
        if _matches_node_text_filters(
            node,
            document_title=document_title,
            search=search,
        )
    ][:limit]

    return {
        "nodes": filtered,
        "facets": _build_node_facets(filtered),
        "total": len(filtered),
    }


@router.get("/collections/{collection_name}/documents")
def list_documents(
    collection_name: str,
    limit: int = Query(200, ge=1, le=500),
    client_id: str | None = Query(None),
    document_title: str | None = Query(None),
    parser_name: str | None = Query(None),
    search: str | None = Query(None),
):
    client = vector_store_manager.get_qdrant_client()
    try:
        if not client.collection_exists(collection_name):
            raise HTTPException(status_code=404, detail="Collection not found")

        scroll_filter = _build_node_scroll_filter(
            client_id=client_id,
            document_id=None,
            parser_name=parser_name,
            chunk_type=None,
            page_num=None,
        )
        records = _scroll_node_records(
            client,
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=DOCUMENT_LIST_SCAN_LIMIT,
            post_filter_enabled=True,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return list_document_candidates(
        collection_name=collection_name,
        records=records,
        serialize_node=lambda name, record: _serialize_node(
            name,
            record,
            include_raw_payload=False,
        ),
        document_title=document_title,
        search=search,
        limit=limit,
    )


@router.get("/collections/{collection_name}/documents/compare")
def compare_documents(
    collection_name: str,
    document_keys: list[str] = Query(..., min_length=DOCUMENT_COMPARE_MIN),
):
    client = vector_store_manager.get_qdrant_client()
    try:
        if not client.collection_exists(collection_name):
            raise HTTPException(status_code=404, detail="Collection not found")
        return compare_qdrant_documents(
            client=client,
            collection_name=collection_name,
            document_keys=document_keys,
            serialize_node=lambda name, record: _serialize_node(
                name,
                record,
                include_raw_payload=False,
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/collections/{collection_name}/nodes/compare")
def compare_nodes(
    collection_name: str,
    point_ids: list[str] = Query(..., min_length=_NODE_COMPARE_MIN),
):
    if len(point_ids) < _NODE_COMPARE_MIN or len(point_ids) > _NODE_COMPARE_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"Select between {_NODE_COMPARE_MIN} and {_NODE_COMPARE_MAX} nodes.",
        )

    client = vector_store_manager.get_qdrant_client()
    try:
        records = client.retrieve(
            collection_name=collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    by_id = {str(getattr(record, "id", "")): record for record in records}
    missing = [point_id for point_id in point_ids if point_id not in by_id]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={"message": "One or more points were not found.", "missing": missing},
        )

    return {
        "nodes": [
            _serialize_node(collection_name, by_id[point_id], include_raw_payload=True)
            for point_id in point_ids
        ]
    }


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


def _serialize_node(
    collection_name: str,
    record: Any,
    *,
    include_raw_payload: bool,
) -> dict[str, Any]:
    payload = getattr(record, "payload", None) or {}
    if not isinstance(payload, dict):
        payload = {}
    node_payload = _node_payload(payload)
    metadata = node_payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    text = _node_text(node_payload, payload)
    text_preview = _text_preview(text)
    page_nums = _int_list(
        _first_value(metadata, payload, "page_nums")
        or _first_value(metadata, payload, "page_num", "page", "slide_num")
    )

    node = {
        "id": str(getattr(record, "id", "")),
        "collection": collection_name,
        "document_id": _first_text(metadata, payload, "document_id", "doc_id"),
        "document_name": _first_text(
            metadata,
            payload,
            "document_name",
            "file_name",
            "filename",
            "source_file",
        ),
        "client_id": _first_text(metadata, payload, "client_id"),
        "parser_name": _first_text(metadata, payload, "parser_name"),
        "parser_version": _first_text(metadata, payload, "parser_version"),
        "chunk_id": _first_text(metadata, payload, "chunk_id"),
        "chunk_type": _first_text(metadata, payload, "chunk_type"),
        "page_num": _first_int(metadata, payload, "page_num", "page", "slide_num"),
        "page_nums": page_nums,
        "section_path": _first_text(metadata, payload, "section_path"),
        "citation_label": _first_text(metadata, payload, "citation_label"),
        "text": text,
        "text_preview": text_preview,
        "text_length": len(text) if isinstance(text, str) else 0,
        "node_metadata": metadata,
        "top_level_metadata": _top_level_metadata(payload),
    }
    if include_raw_payload:
        node["raw_payload"] = payload
    return node


def _build_node_scroll_filter(
    *,
    client_id: str | None,
    document_id: str | None,
    parser_name: str | None,
    chunk_type: str | None,
    page_num: int | None,
) -> Any:
    conditions = []
    filters = {
        "client_id": client_id,
        "document_id": document_id,
        "parser_name": parser_name,
        "chunk_type": chunk_type,
    }
    for key, value in filters.items():
        cleaned = _clean_missing(value)
        if cleaned is None:
            continue
        conditions.append(
            qdrant_models.FieldCondition(
                key=key,
                match=qdrant_models.MatchValue(value=cleaned),
            )
        )

    if page_num is not None:
        conditions.append(
            qdrant_models.FieldCondition(
                key="page_num",
                match=qdrant_models.MatchValue(value=page_num),
            )
        )

    if not conditions:
        return None

    return qdrant_models.Filter(must=conditions)


def _scroll_node_records(
    client: Any,
    *,
    collection_name: str,
    scroll_filter: Any,
    limit: int,
    post_filter_enabled: bool,
) -> list[Any]:
    records: list[Any] = []
    offset = None
    scanned = 0
    max_scan = min(max(limit * (10 if post_filter_enabled else 1), limit), 5000)

    while len(records) < max_scan:
        batch_limit = min(256, max_scan - len(records))
        batch, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch_limit,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        scanned += len(batch or [])
        records.extend(batch or [])
        if offset is None or not batch:
            break
        if not post_filter_enabled and scanned >= limit:
            break

    return records


def _matches_node_text_filters(
    node: dict[str, Any],
    *,
    document_title: str | None,
    search: str | None,
) -> bool:
    title = _clean_missing(document_title)
    if title is not None:
        title_lower = title.lower()
        document_name = str(node.get("document_name") or "").lower()
        if title_lower not in document_name:
            return False

    needle = _clean_missing(search)
    if needle is None:
        return True

    haystack = " ".join(
        str(value or "")
        for value in (
            node.get("document_name"),
            node.get("parser_name"),
            node.get("chunk_type"),
            node.get("chunk_id"),
            node.get("citation_label"),
            node.get("section_path"),
            node.get("text_preview"),
        )
    ).lower()
    return needle.lower() in haystack




def _build_node_facets(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    documents: dict[str, dict[str, Any]] = {}
    parsers: dict[str, int] = {}
    chunk_types: dict[str, int] = {}

    for node in nodes:
        document_label = node.get("document_name") or node.get("document_id")
        if document_label:
            key = str(node.get("document_id") or document_label)
            item = documents.setdefault(
                key,
                {
                    "id": node.get("document_id"),
                    "title": document_label,
                    "count": 0,
                },
            )
            item["count"] += 1

        parser = node.get("parser_name")
        if parser:
            parsers[str(parser)] = parsers.get(str(parser), 0) + 1

        chunk_type = node.get("chunk_type")
        if chunk_type:
            chunk_types[str(chunk_type)] = chunk_types.get(str(chunk_type), 0) + 1

    return {
        "documents": sorted(documents.values(), key=lambda item: str(item["title"])),
        "parsers": [
            {"name": name, "count": count}
            for name, count in sorted(parsers.items())
        ],
        "chunk_types": [
            {"name": name, "count": count}
            for name, count in sorted(chunk_types.items())
        ],
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


def _node_text(node_payload: dict[str, Any], payload: dict[str, Any]) -> str | None:
    text = node_payload.get("text")
    if not isinstance(text, str):
        text_resource = node_payload.get("text_resource")
        if isinstance(text_resource, dict):
            text = text_resource.get("text")
    if not isinstance(text, str):
        text = payload.get("text") or payload.get("chunk_text")
    if isinstance(text, str):
        return text
    return None


def _chunk_preview(node_payload: dict[str, Any], payload: dict[str, Any]) -> str | None:
    return _text_preview(_node_text(node_payload, payload))


def _text_preview(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    return " ".join(text.split())[:_NODE_TEXT_PREVIEW_CHARS]


def _top_level_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "_node_content"}


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
            value = _clean_missing(source.get(key))
            if value is not None:
                return str(value)
    return None


def _first_value(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *keys: str,
) -> Any:
    for source in (primary, secondary):
        for key in keys:
            value = _clean_missing(source.get(key))
            if value is not None:
                return value
    return None


def _first_int(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *keys: str,
) -> int | None:
    return safe_int(_first_text(primary, secondary, *keys))


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, list | tuple | set) else [value]
    result = []
    for item in values:
        parsed = safe_int(item)
        if parsed is not None:
            result.append(parsed)
    return result


def _clean_missing(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in _MISSING_SENTINELS:
            return None
        return normalized
    return value
