from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st
from qdrant_client.http import models as qdrant_models

from app.indexing.vector_store import COLLECTION_NAME, vector_store_manager
from ui.components.layout import render_page_shell
from ui.components.utils import get_client_options

_MISSING_SENTINELS = {"", "none", "null", "n/a", "na", "undefined"}


def render_qdrant_inspector():
    render_page_shell(
        "Inspect Qdrant nodes by document.",
        "Load live points from Qdrant or paste JSON payloads, then review document-level groups and point details.",
        "Qdrant inspector",
        icon="dataset",
    )
    st.caption(f"Collection: `{COLLECTION_NAME}`")
    st.caption(
        "Grouping key order: nested `_node_content.metadata.document_id` -> top-level `document_id` -> document/file name."
    )

    source_mode = st.radio(
        "Source",
        ["Live Qdrant", "Paste JSON"],
        horizontal=True,
        key="qdrant_inspector_source_mode",
    )
    search_term = st.text_input(
        "Document filter (name or id contains)",
        key="qdrant_inspector_document_filter",
        placeholder="e.g. investor presentation or 46882477-...",
    )

    if source_mode == "Live Qdrant":
        points = _render_live_controls()
    else:
        points = _render_json_controls()

    if not points:
        st.info("Load points to inspect document groups.")
        return

    normalized_points = []
    for idx, point in enumerate(points, start=1):
        normalized = _normalize_point_item(point, index=idx)
        if normalized:
            normalized_points.append(normalized)

    if search_term.strip():
        needle = search_term.strip().lower()
        normalized_points = [
            point
            for point in normalized_points
            if needle in point["document_key"].lower()
            or needle in (point.get("document_name") or "").lower()
            or needle in (point.get("document_id") or "").lower()
        ]

    grouped = _group_points_by_document(normalized_points)
    if not grouped:
        st.warning("No points matched the current filters.")
        return

    _render_grouped_results(grouped, normalized_points)


def _render_live_controls() -> list[dict[str, Any]]:
    session_key = "qdrant_inspector_live_points"
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    client_options, client_names = get_client_options()
    choices = ["All clients"] + client_names if client_names else ["All clients"]

    control_cols = st.columns([0.4, 0.2, 0.2, 0.2], vertical_alignment="bottom")
    with control_cols[0]:
        selected_client = st.selectbox(
            "Client",
            choices,
            key="qdrant_inspector_client",
        )
    with control_cols[1]:
        max_points = int(
            st.number_input(
                "Max points",
                min_value=1,
                max_value=5000,
                value=500,
                step=50,
                key="qdrant_inspector_max_points",
            )
        )
    with control_cols[2]:
        batch_size = int(
            st.number_input(
                "Batch size",
                min_value=1,
                max_value=1000,
                value=200,
                step=50,
                key="qdrant_inspector_batch_size",
            )
        )
    with control_cols[3]:
        load_clicked = st.button(
            "Load points",
            type="primary",
            icon=":material/download:",
            key="qdrant_inspector_load_live",
            width="stretch",
        )

    if load_clicked:
        selected_client_id = None
        if selected_client != "All clients":
            selected_client_id = client_options.get(selected_client)
        with st.spinner("Loading points from Qdrant..."):
            try:
                st.session_state[session_key] = _fetch_live_points(
                    max_points=max_points,
                    batch_size=batch_size,
                    client_id=selected_client_id,
                )
                st.success(f"Loaded {len(st.session_state[session_key])} points.")
            except Exception as exc:
                st.error(f"Failed to load Qdrant points: {exc}")

    if st.button(
        "Clear live results",
        icon=":material/close:",
        key="qdrant_inspector_clear_live",
        width="stretch",
    ):
        st.session_state[session_key] = []
        st.rerun()

    return st.session_state.get(session_key, [])


def _render_json_controls() -> list[dict[str, Any]]:
    session_key = "qdrant_inspector_json_points"
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    raw_json = st.text_area(
        "Point JSON",
        key="qdrant_inspector_json_input",
        height=320,
        placeholder="Paste a single point object or a list of points.",
    )
    parse_clicked = st.button(
        "Parse JSON",
        type="primary",
        icon=":material/data_object:",
        key="qdrant_inspector_parse_json",
        width="stretch",
    )

    if parse_clicked:
        try:
            st.session_state[session_key] = _parse_pasted_points(raw_json)
            st.success(f"Parsed {len(st.session_state[session_key])} point objects.")
        except ValueError as exc:
            st.error(str(exc))

    if st.button(
        "Clear pasted results",
        icon=":material/close:",
        key="qdrant_inspector_clear_json",
        width="stretch",
    ):
        st.session_state[session_key] = []
        st.rerun()

    return st.session_state.get(session_key, [])


def _fetch_live_points(
    max_points: int,
    batch_size: int,
    client_id: str | None,
) -> list[dict[str, Any]]:
    qdrant = vector_store_manager.get_qdrant_client()
    if not qdrant.collection_exists(COLLECTION_NAME):
        raise ValueError(f"Collection '{COLLECTION_NAME}' does not exist.")

    scroll_filter = None
    if client_id:
        scroll_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="client_id",
                    match=qdrant_models.MatchValue(value=client_id),
                )
            ]
        )

    all_points: list[dict[str, Any]] = []
    offset = None

    while len(all_points) < max_points:
        limit = min(batch_size, max_points - len(all_points))
        records, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not records:
            break

        for record in records:
            all_points.append({"id": record.id, "payload": record.payload or {}})

        if offset is None:
            break

    return all_points


def _parse_pasted_points(raw_json: str) -> list[dict[str, Any]]:
    if not raw_json.strip():
        return []

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        if isinstance(parsed.get("points"), list):
            items = parsed["points"]
        else:
            items = [parsed]
    else:
        raise ValueError("JSON must be an object or a list of objects.")

    cleaned: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every point entry must be a JSON object.")
        cleaned.append(item)
    return cleaned


def _normalize_point_item(item: dict[str, Any], index: int) -> dict[str, Any] | None:
    payload = item.get("payload")
    if not isinstance(payload, dict):
        payload = item if isinstance(item, dict) else {}
    if not isinstance(payload, dict):
        return None

    node_content = _parse_node_content(payload.get("_node_content"))
    nested_metadata = node_content.get("metadata", {})
    if not isinstance(nested_metadata, dict):
        nested_metadata = {}

    document_id = _first_non_missing(
        nested_metadata.get("document_id"),
        payload.get("document_id"),
    )
    document_name = _first_non_missing(
        nested_metadata.get("document_name"),
        payload.get("document_name"),
        nested_metadata.get("file_name"),
        payload.get("file_name"),
        nested_metadata.get("source_file"),
        payload.get("source_file"),
    )
    document_key = document_id or document_name or "unknown_document"

    point_id = _first_non_missing(
        item.get("id"),
        node_content.get("id_"),
        payload.get("chunk_id"),
    )
    if point_id is None:
        point_id = f"point-{index}"

    page_num = _coerce_int(
        _first_non_missing(nested_metadata.get("page_num"), payload.get("page_num"))
    )
    page_nums = _normalize_page_nums(
        nested_metadata.get("page_nums") or payload.get("page_nums")
    )

    parsed_text = _extract_parsed_text(node_content)
    text_preview = _build_text_preview(parsed_text)
    top_level_metadata = _extract_top_level_metadata(payload)

    return {
        "point_id": str(point_id),
        "document_key": str(document_key),
        "document_id": str(document_id) if document_id is not None else None,
        "document_name": str(document_name) if document_name is not None else None,
        "client_id": _first_non_missing(
            nested_metadata.get("client_id"), payload.get("client_id")
        ),
        "client_name": _first_non_missing(
            nested_metadata.get("client_name"), payload.get("client_name")
        ),
        "chunk_id": _first_non_missing(
            nested_metadata.get("chunk_id"), payload.get("chunk_id")
        ),
        "chunk_type": _first_non_missing(
            nested_metadata.get("chunk_type"), payload.get("chunk_type")
        ),
        "page_num": page_num,
        "page_nums": page_nums,
        "section_path": _first_non_missing(
            nested_metadata.get("section_path"), payload.get("section_path")
        ),
        "citation_label": _first_non_missing(
            nested_metadata.get("citation_label"), payload.get("citation_label")
        ),
        "source_file": _first_non_missing(
            nested_metadata.get("source_file"), payload.get("source_file")
        ),
        "node_type": _first_non_missing(payload.get("_node_type")),
        "parsed_text": parsed_text,
        "node_metadata": dict(nested_metadata),
        "top_level_metadata": top_level_metadata,
        "text_preview": text_preview,
        "raw_payload": payload,
    }


def _group_points_by_document(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}

    for point in points:
        document_key = point["document_key"]
        if document_key not in groups:
            groups[document_key] = {
                "document_key": document_key,
                "document_id": point.get("document_id"),
                "document_name": point.get("document_name"),
                "points": [],
            }

        group = groups[document_key]
        if not group.get("document_id") and point.get("document_id"):
            group["document_id"] = point["document_id"]
        if not group.get("document_name") and point.get("document_name"):
            group["document_name"] = point["document_name"]
        group["points"].append(point)

    result = []
    for key, group in groups.items():
        points_in_group = group["points"]
        page_numbers = {
            page
            for point in points_in_group
            for page in point.get("page_nums", [])
            if page is not None
        }
        for point in points_in_group:
            if point.get("page_num") is not None:
                page_numbers.add(point["page_num"])

        chunk_types = sorted(
            {
                str(chunk_type)
                for chunk_type in (point.get("chunk_type") for point in points_in_group)
                if chunk_type
            }
        )
        client_ids = sorted(
            {
                str(client_id)
                for client_id in (p.get("client_id") for p in points_in_group)
                if client_id
            }
        )

        result.append(
            {
                "document_key": key,
                "document_id": group.get("document_id"),
                "document_name": group.get("document_name"),
                "point_count": len(points_in_group),
                "page_count": len(page_numbers),
                "chunk_types": ", ".join(chunk_types),
                "client_ids": ", ".join(client_ids),
                "points": points_in_group,
            }
        )

    result.sort(key=lambda item: (-item["point_count"], item["document_key"]))
    return result


def _render_grouped_results(
    grouped: list[dict[str, Any]], normalized_points: list[dict[str, Any]]
) -> None:
    total_points = len(normalized_points)
    total_documents = len(grouped)
    average_points = total_points / total_documents if total_documents else 0.0

    metric_cols = st.columns(3)
    metric_cols[0].metric("Total points", f"{total_points}")
    metric_cols[1].metric("Document groups", f"{total_documents}")
    metric_cols[2].metric("Avg points / document", f"{average_points:.2f}")

    summary_rows = [
        {
            "document_key": group["document_key"],
            "document_id": group.get("document_id"),
            "document_name": group.get("document_name"),
            "point_count": group["point_count"],
            "page_count": group["page_count"],
            "chunk_types": group["chunk_types"] or "-",
            "client_ids": group["client_ids"] or "-",
        }
        for group in grouped
    ]
    st.subheader("Document summary")
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    st.download_button(
        "Download grouped JSON",
        data=json.dumps(_serialize_groups(grouped), indent=2),
        file_name="qdrant_grouped_points.json",
        mime="application/json",
        width="stretch",
    )

    st.subheader("Points by document")
    for idx, group in enumerate(grouped):
        display_name = group.get("document_name") or group["document_key"]
        label = f"{display_name} ({group['point_count']} points)"
        with st.expander(label):
            point_rows = [
                {
                    "point_id": point["point_id"],
                    "document_id": point.get("document_id"),
                    "chunk_id": point.get("chunk_id"),
                    "chunk_type": point.get("chunk_type"),
                    "page_num": point.get("page_num"),
                    "section_path": point.get("section_path"),
                    "citation_label": point.get("citation_label"),
                    "text_preview": point.get("text_preview"),
                }
                for point in group["points"]
            ]
            st.dataframe(pd.DataFrame(point_rows), width="stretch", hide_index=True)

            point_options = {
                _point_option_label(point, rank + 1): point
                for rank, point in enumerate(group["points"])
            }
            selected_point_label = st.selectbox(
                "Inspect point",
                list(point_options.keys()),
                key=f"qdrant_point_selector_{idx}",
            )
            selected_point = point_options[selected_point_label]

            metadata_tab, parsed_text_tab = st.tabs(["Metadata", "Parsed text"])

            with metadata_tab:
                st.caption("Node metadata (`_node_content.metadata`)")
                st.json(selected_point.get("node_metadata") or {})
                st.caption("Top-level payload metadata")
                st.json(selected_point.get("top_level_metadata") or {})

            with parsed_text_tab:
                parsed_text = selected_point.get("parsed_text")
                text_len = len(parsed_text) if isinstance(parsed_text, str) else 0
                st.metric("Text length", text_len)
                if parsed_text:
                    st.text_area(
                        "Parsed text",
                        value=parsed_text,
                        height=280,
                        key=f"qdrant_point_text_{idx}",
                    )
                else:
                    st.info("No parsed text found for this point.")

            with st.expander("Raw payload"):
                st.json(selected_point["raw_payload"])


def _serialize_groups(grouped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []
    for group in grouped:
        serialized.append(
            {
                "document_key": group["document_key"],
                "document_id": group.get("document_id"),
                "document_name": group.get("document_name"),
                "point_count": group["point_count"],
                "page_count": group["page_count"],
                "chunk_types": group["chunk_types"],
                "client_ids": group["client_ids"],
                "points": [
                    {
                        "point_id": point["point_id"],
                        "chunk_id": point.get("chunk_id"),
                        "chunk_type": point.get("chunk_type"),
                        "page_num": point.get("page_num"),
                        "page_nums": point.get("page_nums"),
                        "section_path": point.get("section_path"),
                        "citation_label": point.get("citation_label"),
                        "source_file": point.get("source_file"),
                        "parsed_text": point.get("parsed_text"),
                        "node_metadata": point.get("node_metadata"),
                        "top_level_metadata": point.get("top_level_metadata"),
                        "text_preview": point.get("text_preview"),
                    }
                    for point in group["points"]
                ],
            }
        )
    return serialized


def _point_option_label(point: dict[str, Any], rank: int) -> str:
    point_id = point.get("point_id") or f"point-{rank}"
    chunk = point.get("chunk_id") or "unknown-chunk"
    page = point.get("page_num")
    page_label = f"p.{page}" if page is not None else "p.-"
    return f"{rank}. {point_id} | {chunk} | {page_label}"


def _extract_parsed_text(node_content: dict[str, Any]) -> str | None:
    return _first_non_missing(
        node_content.get("text"),
        ((node_content.get("text_resource") or {}).get("text")),
    )


def _extract_top_level_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in payload.items():
        if key == "_node_content":
            continue
        if key in {"document_id", "doc_id", "ref_doc_id"}:
            result[key] = _clean_missing(value)
        else:
            result[key] = value
    return result


def _parse_node_content(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _first_non_missing(*values: Any) -> Any:
    for value in values:
        cleaned = _clean_missing(value)
        if cleaned is not None:
            return cleaned
    return None


def _clean_missing(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in _MISSING_SENTINELS:
            return None
        return normalized
    return value


def _coerce_int(value: Any) -> int | None:
    value = _clean_missing(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_page_nums(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    pages = []
    for item in items:
        page = _coerce_int(item)
        if page is not None:
            pages.append(page)
    return pages


def _build_text_preview(value: Any) -> str | None:
    text = _clean_missing(value)
    if text is None:
        return None
    if not isinstance(text, str):
        return str(text)
    if len(text) <= 160:
        return text
    return f"{text[:157]}..."
