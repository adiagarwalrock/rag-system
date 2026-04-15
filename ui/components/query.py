from pathlib import Path

import streamlit as st

from ui.components.auth import get_api
from ui.components.layout import get_current_user_id, render_page_shell
from ui.components.utils import CLIENTS_CACHE_KEY, bump_cache_revision, get_client_options


def _ensure_chat_history():
    st.session_state.setdefault("chat_history_by_client", {})


def _get_chat_history(client_id: str) -> list[dict]:
    _ensure_chat_history()
    return st.session_state.chat_history_by_client.setdefault(client_id, [])


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


def _submit_question(api, client_id: str, question: str):
    chat_history = _get_chat_history(client_id)
    chat_history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and drafting a sourced answer..."):
            try:
                result = api.query(client_id, question, user_id=get_current_user_id())
                answer = result.get("answer", "No answer generated.")
                streamed_answer = st.write_stream(_stream_text(answer))
                _render_result_details(result)
                chat_history.append(
                    {"role": "assistant", "content": streamed_answer, "result": result}
                )
            except Exception as exc:
                error_msg = f"I could not complete that search: {exc}"
                st.error(error_msg)
                chat_history.append({"role": "assistant", "content": error_msg})


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

    st.session_state.setdefault("query_active_client_name", client_names[0])
    if st.session_state["query_active_client_name"] not in client_names:
        st.session_state["query_active_client_name"] = client_names[0]
    st.session_state.setdefault(
        "query_active_client_id",
        client_options[st.session_state["query_active_client_name"]],
    )
    st.session_state["query_active_client_id"] = client_options[
        st.session_state["query_active_client_name"]
    ]

    with st.sidebar:
        with st.form("query_workspace_form"):
            selected_name = st.selectbox(
                "Client workspace",
                client_names,
                index=max(
                    0,
                    client_names.index(st.session_state["query_active_client_name"])
                    if st.session_state["query_active_client_name"] in client_names
                    else 0,
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
                st.session_state["query_active_client_id"] = client_options[selected_name]
                st.rerun()

        selected_name = st.session_state["query_active_client_name"]
        selected_client_id = st.session_state["query_active_client_id"]

        if st.button(
            "Clear chat",
            icon=":material/delete:",
            key="clear_chat",
            width="stretch",
        ):
            st.session_state.chat_history_by_client[selected_client_id] = []
            st.rerun()

        if st.button(
            "Refresh clients",
            icon=":material/refresh:",
            key="refresh_query_clients",
            width="stretch",
        ):
            bump_cache_revision(CLIENTS_CACHE_KEY)
            st.rerun()

    chat_history = _get_chat_history(selected_client_id)

    for message in chat_history:
        _render_chat_message(message)

    if question := st.chat_input(f"Ask about {selected_name}'s documents"):
        trimmed = question.strip()
        if trimmed:
            _submit_question(api, selected_client_id, trimmed)
