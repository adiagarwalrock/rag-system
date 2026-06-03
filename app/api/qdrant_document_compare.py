from __future__ import annotations

import base64
import json
import re
from collections import Counter
from typing import Any, Callable

from fastapi import HTTPException
from qdrant_client.http import models as qdrant_models

DOCUMENT_COMPARE_MIN = 2
DOCUMENT_COMPARE_MAX = 4
DOCUMENT_LIST_SCAN_LIMIT = 5000
DOCUMENT_COMPARE_SCAN_LIMIT = 10000

SerializeNode = Callable[[str, Any], dict[str, Any]]


def list_document_candidates(
    *,
    collection_name: str,
    records: list[Any],
    serialize_node: SerializeNode,
    document_title: str | None,
    search: str | None,
    limit: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        node = serialize_node(collection_name, record)
        if not _matches_document_filters(
            node,
            document_title=document_title,
            search=search,
        ):
            continue
        groups.setdefault(_document_key_for_node(node), []).append(node)

    documents = [_summarize_candidate(collection_name, nodes) for nodes in groups.values()]
    documents.sort(
        key=lambda item: (
            str(item.get("document_name") or ""),
            str(item.get("client_id") or ""),
            str(item.get("parser_name") or ""),
            str(item.get("parser_version") or ""),
        )
    )
    documents = documents[:limit]
    return {
        "documents": documents,
        "facets": _build_document_facets(documents),
        "total": len(documents),
    }


def compare_documents(
    *,
    client: Any,
    collection_name: str,
    document_keys: list[str],
    serialize_node: SerializeNode,
) -> dict[str, Any]:
    if len(document_keys) < DOCUMENT_COMPARE_MIN or len(document_keys) > DOCUMENT_COMPARE_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"Select between {DOCUMENT_COMPARE_MIN} and {DOCUMENT_COMPARE_MAX} documents.",
        )

    selectors = [_decode_document_key(key) for key in document_keys]
    documents = []
    missing: list[str] = []

    for raw_key, selector in zip(document_keys, selectors):
        records = _scroll_records(
            client,
            collection_name=collection_name,
            scroll_filter=_build_document_scroll_filter(selector),
            max_records=DOCUMENT_COMPARE_SCAN_LIMIT,
        )
        nodes = [
            serialize_node(collection_name, record)
            for record in records
        ]
        matched_nodes = [
            node for node in nodes if _document_identity(node) == _selector_identity(selector)
        ]
        if not matched_nodes:
            missing.append(raw_key)
            continue
        documents.append(_build_document_compare(collection_name, matched_nodes))

    if missing:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "One or more documents were not found.",
                "missing": missing,
            },
        )

    return {"documents": documents}


def _scroll_records(
    client: Any,
    *,
    collection_name: str,
    scroll_filter: Any,
    max_records: int,
) -> list[Any]:
    records: list[Any] = []
    offset = None

    while len(records) < max_records:
        batch_limit = min(256, max_records - len(records))
        batch, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch_limit,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        records.extend(batch or [])
        if offset is None or not batch:
            break

    return records


def _build_document_scroll_filter(selector: dict[str, str | None]) -> Any:
    conditions = []
    # Qdrant requires payload indexes for filtered fields. Keep the server-side
    # filter on indexed coarse identity fields, then enforce the full opaque
    # document key, including parser_version, in the Python post-filter.
    filter_keys = ["client_id", "document_id", "parser_name"]
    if not selector.get("document_id"):
        filter_keys.append("document_name")

    for key in filter_keys:
        value = selector.get(key)
        if value:
            conditions.append(
                qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchValue(value=value),
                )
            )

    if not conditions:
        return None
    return qdrant_models.Filter(must=conditions)


def _build_document_compare(
    collection_name: str,
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered_nodes = sorted(nodes, key=_node_sort_key)
    candidate = _summarize_candidate(collection_name, ordered_nodes)
    markdown = "\n\n".join(
        text.strip()
        for text in (node.get("text") for node in ordered_nodes)
        if isinstance(text, str) and text.strip()
    )
    pages = _document_pages(ordered_nodes)
    chunk_types = Counter(
        str(node.get("chunk_type"))
        for node in ordered_nodes
        if node.get("chunk_type")
    )

    return {
        **candidate,
        "markdown": markdown,
        "text_length": len(markdown),
        "metadata_summary": {
            "pages": pages,
            "chunk_types": dict(sorted(chunk_types.items())),
            "source_node_count": len(ordered_nodes),
        },
        "source_nodes": [
            {
                "id": node.get("id"),
                "chunk_id": node.get("chunk_id"),
                "chunk_type": node.get("chunk_type"),
                "page_num": node.get("page_num"),
                "page_nums": node.get("page_nums") or [],
                "section_path": node.get("section_path"),
                "citation_label": node.get("citation_label"),
                "text_length": node.get("text_length") or 0,
                "text_preview": node.get("text_preview"),
            }
            for node in ordered_nodes
        ],
    }


def _summarize_candidate(
    collection_name: str,
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered_nodes = sorted(nodes, key=_node_sort_key)
    first = ordered_nodes[0]
    text_parts = [
        text.strip()
        for text in (node.get("text") for node in ordered_nodes)
        if isinstance(text, str) and text.strip()
    ]
    markdown = "\n\n".join(text_parts)
    pages = _document_pages(ordered_nodes)

    return {
        "document_key": _document_key_for_node(first),
        "collection": collection_name,
        "client_id": first.get("client_id"),
        "document_id": first.get("document_id"),
        "document_name": first.get("document_name"),
        "parser_name": first.get("parser_name"),
        "parser_version": first.get("parser_version"),
        "node_count": len(ordered_nodes),
        "page_count": len(pages),
        "text_length": len(markdown),
        "preview": _preview(markdown),
    }


def _build_document_facets(
    documents: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    titles: dict[str, dict[str, Any]] = {}
    clients: dict[str, int] = {}
    parsers: dict[str, int] = {}

    for document in documents:
        title = document.get("document_name") or document.get("document_id")
        if title:
            key = str(document.get("document_id") or title)
            item = titles.setdefault(
                key,
                {
                    "id": document.get("document_id"),
                    "title": str(title),
                    "count": 0,
                },
            )
            item["count"] += 1

        client_id = document.get("client_id")
        if client_id:
            clients[str(client_id)] = clients.get(str(client_id), 0) + 1

        parser = document.get("parser_name")
        if parser:
            parsers[str(parser)] = parsers.get(str(parser), 0) + 1

    return {
        "documents": sorted(titles.values(), key=lambda item: str(item["title"])),
        "clients": [
            {"id": client_id, "count": count}
            for client_id, count in sorted(clients.items())
        ],
        "parsers": [
            {"name": name, "count": count}
            for name, count in sorted(parsers.items())
        ],
    }


def _matches_document_filters(
    node: dict[str, Any],
    *,
    document_title: str | None,
    search: str | None,
) -> bool:
    title = _clean(document_title)
    if title:
        title_lower = title.lower()
        document_name = str(node.get("document_name") or "").lower()
        document_id = str(node.get("document_id") or "").lower()
        if title_lower not in document_name and title_lower not in document_id:
            return False

    needle = _clean(search)
    if not needle:
        return True

    needle_lower = needle.lower()
    haystack = " ".join(
        str(value or "")
        for value in (
            node.get("document_name"),
            node.get("document_id"),
            node.get("client_id"),
            node.get("parser_name"),
            node.get("parser_version"),
            node.get("chunk_type"),
            node.get("citation_label"),
            node.get("section_path"),
            node.get("text_preview"),
        )
    ).lower()
    return needle_lower in haystack


def _document_key_for_node(node: dict[str, Any]) -> str:
    payload = {
        "client_id": _clean(node.get("client_id")),
        "document_id": _clean(node.get("document_id")),
        "document_name": _clean(node.get("document_name")),
        "parser_name": _clean(node.get("parser_name")),
        "parser_version": _clean(node.get("parser_version")),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return encoded.rstrip(b"=").decode("ascii")


def _decode_document_key(document_key: str) -> dict[str, str | None]:
    try:
        padding = "=" * (-len(document_key) % 4)
        raw = base64.urlsafe_b64decode(f"{document_key}{padding}".encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid document key.") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Invalid document key.")

    selector = {
        key: _clean(payload.get(key))
        for key in (
            "client_id",
            "document_id",
            "document_name",
            "parser_name",
            "parser_version",
        )
    }
    if not selector.get("document_id") and not selector.get("document_name"):
        raise HTTPException(status_code=422, detail="Invalid document key.")
    return selector


def _document_identity(node: dict[str, Any]) -> tuple[str | None, ...]:
    return (
        _clean(node.get("client_id")),
        _clean(node.get("document_id")),
        _clean(node.get("document_name")),
        _clean(node.get("parser_name")),
        _clean(node.get("parser_version")),
    )


def _selector_identity(selector: dict[str, str | None]) -> tuple[str | None, ...]:
    return (
        selector.get("client_id"),
        selector.get("document_id"),
        selector.get("document_name"),
        selector.get("parser_name"),
        selector.get("parser_version"),
    )


def _document_pages(nodes: list[dict[str, Any]]) -> list[int]:
    pages: set[int] = set()
    for node in nodes:
        for page in node.get("page_nums") or []:
            if isinstance(page, int):
                pages.add(page)
        page_num = node.get("page_num")
        if isinstance(page_num, int):
            pages.add(page_num)
    return sorted(pages)


def _node_sort_key(node: dict[str, Any]) -> tuple[int, tuple[Any, ...], str, str]:
    pages = _document_pages([node])
    page = pages[0] if pages else 1_000_000
    return (
        page,
        _natural_sort_key(str(node.get("chunk_id") or "")),
        str(node.get("section_path") or ""),
        str(node.get("id") or ""),
    )


def _natural_sort_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _preview(text: str) -> str | None:
    if not text:
        return None
    return " ".join(text.split())[:500]


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return str(value)
