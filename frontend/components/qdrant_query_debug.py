import streamlit as st

from frontend.components.api_client import get_api
from frontend.components.layout import render_page_shell


def render_qdrant_query_debug() -> None:
    render_page_shell(
        "Qdrant Query Debug",
        (
            "Run a query through the retrieval pipeline and inspect raw chunks "
            "without LLM synthesis."
        ),
        label="Qdrant Query Debug",
        icon="manage_search",
    )

    api = get_api()

    # --- Client selector ---
    try:
        clients = api.list_clients()
    except Exception as e:
        st.error(f"Could not load clients: {e}")
        return

    if not clients:
        st.info("No clients found. Create a client first.")
        return

    client_options = {c["name"]: c["id"] for c in clients}
    selected_name = st.selectbox("Client", options=list(client_options.keys()))
    client_id = client_options[selected_name]

    # --- Query inputs ---
    question = st.text_area(
        "Query",
        placeholder="Enter your query here...",
        height=100,
    )
    top_k = st.number_input(
        "Top-K results",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )
    rerank = st.toggle(
        "Rerank results",
        value=True,
        help="Apply semantic and temporal reranking. Disable to see raw Qdrant order.",
    )

    run = st.button("Retrieve", type="primary", disabled=not question.strip())

    if not run:
        return

    with st.spinner("Retrieving..."):
        try:
            results = api.retrieve_only(
                client_id=client_id,
                question=question.strip(),
                top_k=int(top_k),
                rerank=rerank,
            )
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            return

    if not results:
        st.warning("No results returned.")
        return

    st.success(f"{len(results)} chunk(s) retrieved.")
    st.divider()

    for node in results:
        rank = node["rank"]
        score = node.get("score")
        meta = node.get("metadata", {})
        doc_name = (
            meta.get("document_name")
            or meta.get("file_name")
            or "Unknown document"
        )
        page_num = meta.get("page_num")
        chunk_type = meta.get("chunk_type", "")

        score_str = f"{score:.4f}" if score is not None else "N/A"
        page_str = f"  |  p.{page_num}" if page_num is not None else ""
        type_str = f"  |  {chunk_type}" if chunk_type else ""
        header = (
            f"**#{rank}** - score `{score_str}`  |  "
            f"{doc_name}{page_str}{type_str}"
        )

        with st.expander(header, expanded=(rank == 1)):
            st.text_area(
                "Chunk text",
                value=node.get("text", ""),
                height=180,
                disabled=True,
                key=f"text_{rank}_{node.get('node_id', rank)}",
            )

            flat_meta = {k: str(v) if v is not None else "" for k, v in meta.items()}
            flat_meta["node_id"] = node.get("node_id", "")
            flat_meta["score"] = str(score)

            st.dataframe(
                [{"field": k, "value": v} for k, v in flat_meta.items()],
                width="stretch",
                hide_index=True,
            )
