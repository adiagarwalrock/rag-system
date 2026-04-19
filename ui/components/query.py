from pathlib import Path
from datetime import datetime

import streamlit as st

from ui.components.api_client import get_api
from ui.components.layout import render_page_shell
from ui.components.utils import (
    CLIENTS_CACHE_KEY,
    bump_cache_revision,
    get_client_options,
)

REASONING_EFFORT_OPTIONS = ("low", "medium", "high")


def _stream_text(text: str):
    for token in text.split(" "):
        yield token + " "


def _truncate_text(text: str, limit: int = 120) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3].rstrip()}..."


def _badge_rows(items: list[tuple[str, str, str]], per_row: int = 4):
    if not items:
        return

    for start in range(0, len(items), per_row):
        row_items = items[start : start + per_row]
        cols = st.columns(len(row_items))
        for col, (label, icon, color) in zip(cols, row_items):
            col.badge(label, icon=icon, color=color)


def _latency_color(latency_ms: int | None) -> str:
    if latency_ms is None:
        return "gray"
    if latency_ms < 2500:
        return "green"
    if latency_ms < 8000:
        return "blue"
    if latency_ms < 15000:
        return "orange"
    return "red"


def _conflict_color(conflict_count: int) -> str:
    if conflict_count == 0:
        return "green"
    if conflict_count < 3:
        return "orange"
    return "red"


def _format_reference(citation: dict) -> str | None:
    if citation.get("page_num"):
        return f"Page {citation['page_num']}"
    if citation.get("slide_num"):
        return f"Slide {citation['slide_num']}"
    if citation.get("section_title"):
        return f"Section {citation['section_title']}"
    return None


def _displayable_image_refs(raw_refs) -> list[str]:
    if isinstance(raw_refs, str):
        refs = [raw_refs]
    elif isinstance(raw_refs, list):
        refs = raw_refs
    else:
        return []

    displayable: list[str] = []
    valid_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    for ref in refs:
        if not isinstance(ref, str):
            continue
        candidate = Path(ref).expanduser()
        if not candidate.exists() or candidate.suffix.lower() not in valid_suffixes:
            continue
        displayable.append(str(candidate))
    return displayable


def _render_result_details(result: dict):
    citations = result.get("citations", [])
    conflicts = result.get("conflicts", [])
    source_count = result.get("source_count", len(citations))
    evidence_count = result.get("evidence_count", len(citations))
    image_evidence_count = result.get("image_evidence_count", 0)
    latency_ms = result.get("latency_ms")

    summary_badges = [
        (
            f"Sources {source_count}",
            ":material/source:",
            "blue" if source_count else "gray",
        ),
        (
            f"Evidence {evidence_count}",
            ":material/format_quote:",
            "blue" if evidence_count else "gray",
        ),
        (
            f"Latency {latency_ms if latency_ms is not None else '?'} ms",
            ":material/speed:",
            _latency_color(latency_ms),
        ),
        (
            f"Conflicts {len(conflicts)}",
            ":material/warning:",
            _conflict_color(len(conflicts)),
        ),
    ]

    if retrieval_mode := result.get("retrieval_mode"):
        retrieval_label = (
            "Retrieval hybrid"
            if retrieval_mode == "hybrid"
            else "Retrieval dense fallback"
        )
        summary_badges.append((retrieval_label, ":material/tune:", "gray"))

    if result.get("query_expanded"):
        summary_badges.append(("Query expansion on", ":material/swap_horiz:", "gray"))

    if image_evidence_count:
        summary_badges.append(
            (
                f"Image evidence {image_evidence_count}",
                ":material/image:",
                "gray",
            )
        )

    reasoning_effort = result.get("reasoning_effort")
    if reasoning_effort:
        effort_applied = bool(result.get("reasoning_effort_applied"))
        summary_badges.append(
            (
                (
                    f"Reasoning {reasoning_effort}"
                    if effort_applied
                    else f"Reasoning {reasoning_effort} (not applied)"
                ),
                ":material/psychology:",
                "gray",
            )
        )

    _badge_rows(summary_badges, per_row=4)

    reasoning = (result.get("reasoning") or "").strip()
    if reasoning:
        with st.expander(":material/psychology: Reasoning trace", expanded=False):
            st.write(reasoning)

    if citations:
        with st.expander(
            f":material/article: Sources and citations ({len(citations)})",
            expanded=False,
        ):
            for i, citation in enumerate(citations, 1):
                label = citation.get("citation_label") or citation.get(
                    "document_name", f"Source {i}"
                )
                title = f"{i}. {_truncate_text(label, 90)}"

                with st.expander(title, expanded=False, icon=":material/description:"):
                    st.caption(citation.get("document_name", "Unknown document"))
                    st.write(citation.get("text", "No source text returned."))

                    image_refs = _displayable_image_refs(citation.get("asset_refs"))
                    if image_refs:
                        captions = [
                            f"Evidence image {idx}"
                            for idx, _ in enumerate(image_refs, 1)
                        ]
                        st.image(image_refs, caption=captions, width="content")

                    detail_badges: list[tuple[str, str, str]] = []
                    score = citation.get("score")
                    score_label = (
                        f"Score {score:.3f}"
                        if isinstance(score, (int, float))
                        else "Score -"
                    )
                    detail_badges.append((score_label, ":material/star:", "gray"))

                    if chunk_type := citation.get("chunk_type"):
                        detail_badges.append(
                            (f"Type {chunk_type}", ":material/category:", "gray")
                        )

                    if image_refs:
                        detail_badges.append(
                            (
                                f"Images {len(image_refs)}",
                                ":material/image:",
                                "gray",
                            )
                        )

                    if reference := _format_reference(citation):
                        detail_badges.append((reference, ":material/bookmark:", "gray"))

                    if version := citation.get("version_label"):
                        detail_badges.append(
                            (f"Version {version}", ":material/history:", "gray")
                        )

                    authority = citation.get("authority_score")
                    if authority is not None:
                        detail_badges.append(
                            (
                                (
                                    f"Authority {authority:.2f}"
                                    if isinstance(authority, (int, float))
                                    else f"Authority {authority}"
                                ),
                                ":material/verified_user:",
                                "gray",
                            )
                        )

                    _badge_rows(detail_badges, per_row=3)

    if conflicts:
        with st.expander(
            f":material/warning: Conflict checks ({len(conflicts)})",
            expanded=False,
        ):
            for i, conflict in enumerate(conflicts, 1):
                summary = conflict.get("summary") or f"Conflict {i}"
                with st.expander(
                    f"{i}. {_truncate_text(summary, 110)}",
                    expanded=False,
                    icon=":material/report:",
                ):
                    st.write(summary)

                    for j, chunk in enumerate(
                        conflict.get("supporting_chunks") or [], 1
                    ):
                        doc = chunk.get("document_name", "Unknown document")
                        version = chunk.get("version_label") or "unknown"
                        st.markdown(f"**Source {j}: {doc}**")
                        _badge_rows(
                            [
                                (
                                    f"Version {version}",
                                    ":material/history:",
                                    "gray",
                                )
                            ],
                            per_row=1,
                        )
                        if snippet := chunk.get("text_snippet"):
                            st.caption(snippet)

    if qid := result.get("query_id"):
        st.caption(f":material/fingerprint: Query ID: {qid}")


def _render_chat_message(message: dict):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if result := message.get("result"):
            _render_result_details(result)


def _submit_question(
    api,
    client_id: str,
    question: str,
    reasoning_effort: str,
    session_id: str | None,
) -> dict:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and drafting a sourced answer..."):
            result = api.query(
                client_id,
                question,
                reasoning_effort=reasoning_effort,
                session_id=session_id,
            )
            answer = result.get("answer", "No answer generated.")
            st.write_stream(_stream_text(answer))
            _render_result_details(result)
            return result


def _format_session_timestamp(raw_value: str | None) -> str:
    if not raw_value:
        return "n/a"
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return raw_value


def _session_label(session: dict) -> str:
    title = session.get("title") or "Untitled session"
    timestamp = _format_session_timestamp(session.get("last_activity_at"))
    return f"{title} · {timestamp}"


def _active_session_key(client_id: str) -> str:
    return f"query_active_session_{client_id}"


def _resolve_active_session_id(client_id: str, sessions: list[dict]) -> str | None:
    key = _active_session_key(client_id)
    available_ids = [session["id"] for session in sessions]
    active_id = st.session_state.get(key)
    if active_id not in available_ids:
        active_id = available_ids[0] if available_ids else None
        st.session_state[key] = active_id
    return active_id


def render_query():
    render_page_shell(
        "Ask your client documents.",
        "Each answer is scoped to one client and includes source evidence, latency, and conflict checks.",
        "Chat",
        icon="chat",
    )

    api = get_api()
    client_options, client_names = get_client_options()

    if not client_names:
        st.info("Create a client before starting a chat.")
        return

    active_name = st.session_state.get("query_active_client_name")
    if active_name not in client_names:
        active_name = client_names[0]
        st.session_state["query_active_client_name"] = active_name

    st.session_state["query_active_client_id"] = client_options[active_name]
    st.session_state.setdefault("query_reasoning_effort", "medium")

    with st.sidebar:
        with st.form("query_workspace_form"):
            selected_name = st.selectbox(
                "Client workspace",
                client_names,
                index=max(
                    0,
                    (
                        client_names.index(st.session_state["query_active_client_name"])
                        if st.session_state["query_active_client_name"] in client_names
                        else 0
                    ),
                ),
                help="Every answer is limited to documents for this client.",
            )
            if st.form_submit_button(
                "Apply workspace",
                icon=":material/check:",
                type="primary",
                width="stretch",
            ):
                st.session_state["query_active_client_name"] = selected_name
                st.session_state["query_active_client_id"] = client_options[
                    selected_name
                ]
                st.rerun()

        selected_name = st.session_state["query_active_client_name"]
        selected_client_id = st.session_state["query_active_client_id"]
        sessions = api.list_chat_sessions(selected_client_id, limit=100)
        active_session_id = _resolve_active_session_id(selected_client_id, sessions)

        if sessions:
            session_ids = [session["id"] for session in sessions]
            selected_session_id = st.selectbox(
                "Session",
                session_ids,
                index=max(
                    0,
                    (
                        session_ids.index(active_session_id)
                        if active_session_id in session_ids
                        else 0
                    ),
                ),
                format_func=lambda sid: _session_label(
                    next(session for session in sessions if session["id"] == sid)
                ),
                help="Sessions are isolated by client. Semantic memory still draws relevant context from other sessions.",
            )
            st.session_state[_active_session_key(selected_client_id)] = (
                selected_session_id
            )
            active_session_id = selected_session_id
        else:
            st.caption("No session yet. Ask a question to start one automatically.")

        selected_reasoning_effort = st.selectbox(
            "Reasoning effort",
            REASONING_EFFORT_OPTIONS,
            index=max(
                0,
                (
                    REASONING_EFFORT_OPTIONS.index(
                        st.session_state.get("query_reasoning_effort", "medium")
                    )
                    if st.session_state.get("query_reasoning_effort", "medium")
                    in REASONING_EFFORT_OPTIONS
                    else 1
                ),
            ),
            help="Controls response depth. Applied when OpenAI Responses mode is enabled.",
        )
        st.session_state["query_reasoning_effort"] = selected_reasoning_effort

        if st.button(
            "New session",
            icon=":material/add_circle:",
            key="new_chat_session",
            width="stretch",
        ):
            created_session = api.create_chat_session(selected_client_id)
            st.session_state[_active_session_key(selected_client_id)] = created_session[
                "id"
            ]
            st.rerun()

        if st.button(
            "Clear session",
            icon=":material/delete:",
            key="clear_chat",
            width="stretch",
            disabled=not bool(active_session_id),
        ):
            api.clear_chat_session(active_session_id)
            st.rerun()

        if st.button(
            "Refresh clients",
            icon=":material/refresh:",
            key="refresh_query_clients",
            width="stretch",
        ):
            bump_cache_revision(CLIENTS_CACHE_KEY)
            st.rerun()

    sessions = api.list_chat_sessions(selected_client_id, limit=100)
    active_session_id = _resolve_active_session_id(selected_client_id, sessions)
    chat_history = (
        api.list_chat_messages(active_session_id, limit=500)
        if active_session_id
        else []
    )
    for message in chat_history:
        _render_chat_message(message)

    if question := st.chat_input(f"Ask about {selected_name}'s documents"):
        trimmed = question.strip()
        if trimmed:
            try:
                result = _submit_question(
                    api,
                    selected_client_id,
                    trimmed,
                    st.session_state.get("query_reasoning_effort", "medium"),
                    active_session_id,
                )
                returned_session_id = result.get("session_id")
                if returned_session_id:
                    st.session_state[_active_session_key(selected_client_id)] = (
                        returned_session_id
                    )
            except Exception as exc:
                st.error(str(exc))
