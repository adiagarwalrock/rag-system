from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from sqlalchemy import func

from app.db.models.document import (
    ConflictLog,
    Document,
    DocumentVersion,
    IngestionJob,
    QueryLog,
    RetrievalLog,
    VectorNodeRegistry,
)
from app.db.snowflake import SessionLocal
from app.evals import (
    IngestionEvalCase,
    RetrievalEvalCase,
    build_quality_payload,
    evaluate_ingestion_cases,
    evaluate_retrieval_cases,
)
from ui.components.layout import render_page_shell
from ui.components.utils import get_client_options

DEFAULT_RETRIEVAL_CASES = [
    {
        "query": "What changed in the latest policy version?",
        "relevant_ids": ["chunk-1"],
        "retrieved": [
            {
                "vector_node_id": "chunk-1",
                "citation_label": "Policy v2 p.1",
                "version_label": "v2",
            },
            {
                "vector_node_id": "chunk-2",
                "citation_label": "Policy v1 p.4",
                "version_label": "v1",
            },
        ],
        "expected_version_label": "v2",
        "expect_conflict": False,
        "expected_citation_labels": ["Policy v2 p.1"],
    },
]

DEFAULT_INGESTION_CASES = [
    {
        "document_id": "doc-1",
        "parse_success": True,
        "chunk_count": 4,
        "vector_node_count": 4,
        "required_metadata_fields": [
            "document_id",
            "client_id",
            "file_name",
            "version_label",
            "citation_label",
        ],
        "metadata_by_chunk": [
            {
                "document_id": "doc-1",
                "client_id": "client-1",
                "file_name": "policy.pdf",
                "version_label": "v2",
                "citation_label": "Policy v2 p.1",
            }
        ],
        "indexing_success": True,
    }
]


def render_quality():
    render_page_shell(
        "Measure retrieval and ingestion quality.",
        "Use benchmark evals for recall and attribution, and live telemetry for latency, conflicts, and ingestion health.",
        "Quality",
        icon="monitoring",
    )

    client_options, client_names = get_client_options()
    selected_client_id = None
    selected_client_name = "All clients"

    if client_names:
        selected_client_name = st.selectbox(
            "Client workspace",
            ["All clients"] + client_names,
            index=0,
            help="Filter live telemetry to one client or review all clients together.",
        )
        if selected_client_name != "All clients":
            selected_client_id = client_options[selected_client_name]
    else:
        st.info(
            "Create a client to see live telemetry. Benchmark evals still work without one."
        )

    tabs = st.tabs(["Benchmark evals", "Live telemetry"])

    with tabs[0]:
        _render_benchmark_eval_tab()

    with tabs[1]:
        _render_live_telemetry_tab(selected_client_id, selected_client_name)


def _render_benchmark_eval_tab():
    st.subheader("Benchmark evals")
    st.write(
        "Paste structured retrieval and ingestion cases, or start from the sample payloads below."
    )

    col_left, col_right = st.columns(2)
    with col_left:
        retrieval_text = st.text_area(
            "Retrieval cases JSON",
            value=json.dumps(DEFAULT_RETRIEVAL_CASES, indent=2),
            height=320,
        )
    with col_right:
        ingestion_text = st.text_area(
            "Ingestion cases JSON",
            value=json.dumps(DEFAULT_INGESTION_CASES, indent=2),
            height=320,
        )

    run = st.button("Run evals", type="primary", width="stretch")
    if not run:
        st.caption("Run the evals to see metrics and a downloadable quality payload.")
        return

    retrieval_cases = _parse_retrieval_cases(retrieval_text)
    ingestion_cases = _parse_ingestion_cases(ingestion_text)
    if retrieval_cases is None or ingestion_cases is None:
        return

    retrieval_summary = evaluate_retrieval_cases(retrieval_cases)
    ingestion_summary = evaluate_ingestion_cases(ingestion_cases)
    payload = build_quality_payload(retrieval_summary, ingestion_summary)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Retrieval MRR", f"{retrieval_summary.mean_mrr:.3f}")
    metric_cols[1].metric(
        "Retrieval recall@5", f"{retrieval_summary.mean_recall_at_5:.3f}"
    )
    metric_cols[2].metric(
        "Ingestion parse success", f"{ingestion_summary.parse_success_rate:.3f}"
    )
    metric_cols[3].metric(
        "Metadata completeness", f"{ingestion_summary.avg_metadata_completeness:.3f}"
    )

    detail_cols = st.columns(3)
    detail_cols[0].metric(
        "Version attribution", f"{retrieval_summary.mean_version_attribution:.3f}"
    )
    detail_cols[1].metric(
        "Conflict attribution", f"{retrieval_summary.mean_conflict_attribution:.3f}"
    )
    detail_cols[2].metric(
        "Citation precision", f"{retrieval_summary.mean_citation_precision:.3f}"
    )

    st.divider()
    st.subheader("Retrieval case drill-down")
    st.dataframe(
        pd.DataFrame([asdict(case) for case in retrieval_summary.cases]),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Ingestion case drill-down")
    st.dataframe(
        pd.DataFrame([asdict(case) for case in ingestion_summary.cases]),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Quality payload")
    st.json(payload)
    st.download_button(
        "Download quality JSON",
        data=json.dumps(payload, indent=2, default=str),
        file_name="rag_quality_payload.json",
        mime="application/json",
        width="stretch",
    )


def _render_live_telemetry_tab(client_id: str | None, client_name: str):
    st.subheader("Live telemetry")
    st.write(
        "These are operational metrics from the local database, not gold-labeled eval scores."
    )

    with SessionLocal() as db:
        query_base = db.query(QueryLog)
        ingestion_base = (
            db.query(Document, DocumentVersion, IngestionJob)
            .join(
                DocumentVersion,
                DocumentVersion.document_id == Document.id,
                isouter=True,
            )
            .join(
                IngestionJob,
                IngestionJob.document_id == Document.id,
                isouter=True,
            )
        )

        if client_id:
            query_base = query_base.filter(QueryLog.client_id == client_id)
            ingestion_base = ingestion_base.filter(Document.client_id == client_id)

        query_count = query_base.count()
        avg_latency = query_base.with_entities(func.avg(QueryLog.latency_ms)).scalar()
        conflict_queries = db.query(
            func.count(func.distinct(ConflictLog.query_log_id))
        ).join(QueryLog, QueryLog.id == ConflictLog.query_log_id)
        retrieval_rows = db.query(func.count(RetrievalLog.id)).join(
            QueryLog, QueryLog.id == RetrievalLog.query_log_id
        )
        ingestion_count = db.query(func.count(Document.id))
        completed_ingestions = db.query(func.count(IngestionJob.id)).filter(
            IngestionJob.status == "completed"
        )
        failed_ingestions = db.query(func.count(IngestionJob.id)).filter(
            IngestionJob.status == "failed"
        )
        versioned_docs = db.query(func.count(DocumentVersion.id)).join(
            Document, Document.id == DocumentVersion.document_id
        )
        active_nodes = db.query(func.count(VectorNodeRegistry.id)).filter(
            VectorNodeRegistry.is_active == True
        )

        if client_id:
            conflict_queries = conflict_queries.filter(QueryLog.client_id == client_id)
            retrieval_rows = retrieval_rows.filter(QueryLog.client_id == client_id)
            ingestion_count = ingestion_count.filter(Document.client_id == client_id)
            completed_ingestions = completed_ingestions.filter(
                IngestionJob.client_id == client_id
            )
            failed_ingestions = failed_ingestions.filter(
                IngestionJob.client_id == client_id
            )
            versioned_docs = versioned_docs.filter(Document.client_id == client_id)
            active_nodes = active_nodes.filter(
                VectorNodeRegistry.client_id == client_id
            )

        conflict_queries = conflict_queries.scalar() or 0
        retrieval_rows = retrieval_rows.scalar() or 0
        ingestion_count = ingestion_count.scalar() or 0
        completed_ingestions = completed_ingestions.scalar() or 0
        failed_ingestions = failed_ingestions.scalar() or 0
        versioned_docs = versioned_docs.scalar() or 0
        active_nodes = active_nodes.scalar() or 0

        recent_queries = (
            db.query(
                QueryLog.created_at,
                QueryLog.question,
                QueryLog.status,
                QueryLog.latency_ms,
                func.count(func.distinct(RetrievalLog.id)).label("retrieval_items"),
                func.count(func.distinct(ConflictLog.id)).label("conflicts"),
            )
            .outerjoin(RetrievalLog, RetrievalLog.query_log_id == QueryLog.id)
            .outerjoin(ConflictLog, ConflictLog.query_log_id == QueryLog.id)
        )
        if client_id:
            recent_queries = recent_queries.filter(QueryLog.client_id == client_id)
        recent_queries = (
            recent_queries.group_by(
                QueryLog.id,
                QueryLog.created_at,
                QueryLog.question,
                QueryLog.status,
                QueryLog.latency_ms,
            )
            .order_by(QueryLog.created_at.desc())
            .limit(10)
            .all()
        )
        recent_ingestions = (
            ingestion_base.order_by(Document.created_at.desc()).limit(10).all()
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Queries", f"{query_count}")
    metric_cols[1].metric(
        "Avg latency",
        f"{(avg_latency or 0.0):.0f} ms" if query_count else "n/a",
    )
    metric_cols[2].metric("Queries with conflicts", f"{conflict_queries}")
    metric_cols[3].metric("Retrieved rows", f"{retrieval_rows}")

    metric_cols_2 = st.columns(4)
    metric_cols_2[0].metric("Documents", f"{ingestion_count}")
    metric_cols_2[1].metric("Completed ingestions", f"{completed_ingestions}")
    metric_cols_2[2].metric("Failed ingestions", f"{failed_ingestions}")
    metric_cols_2[3].metric("Active vector nodes", f"{active_nodes}")

    summary_cols = st.columns(2)
    summary_cols[0].metric("Versioned docs", f"{versioned_docs}")
    summary_cols[1].metric(
        "Conflict rate",
        f"{(conflict_queries / query_count):.3f}" if query_count else "n/a",
    )

    st.divider()
    st.subheader("Recent queries")
    if recent_queries:
        query_rows = []
        for row in recent_queries:
            query_rows.append(
                {
                    "created_at": _format_dt(row.created_at),
                    "question": row.question,
                    "status": row.status,
                    "latency_ms": row.latency_ms,
                    "retrieval_items": int(row.retrieval_items or 0),
                    "conflicts": int(row.conflicts or 0),
                }
            )
        st.dataframe(pd.DataFrame(query_rows), width="stretch", hide_index=True)
    else:
        st.caption("No queries recorded yet.")

    st.subheader("Recent ingestions")
    if recent_ingestions:
        ingestion_rows = []
        for document, version, job in recent_ingestions:
            ingestion_rows.append(
                {
                    "created_at": _format_dt(document.created_at),
                    "document": document.name,
                    "status": document.status,
                    "version_label": version.version_label if version else None,
                    "version_group": version.version_group if version else None,
                    "job_status": job.status if job else None,
                    "active_vector_nodes": _active_nodes_for_document(
                        document.id, client_id
                    ),
                }
            )
        st.dataframe(pd.DataFrame(ingestion_rows), width="stretch", hide_index=True)
    else:
        st.caption("No ingestions recorded yet.")


def _parse_retrieval_cases(text: str) -> list[RetrievalEvalCase] | None:
    try:
        items = json.loads(text)
    except json.JSONDecodeError as exc:
        st.error(f"Retrieval cases JSON is invalid: {exc}")
        return None

    try:
        return [
            RetrievalEvalCase(
                query=item["query"],
                relevant_ids=item.get("relevant_ids", []),
                retrieved=item.get("retrieved", []),
                expected_version_label=item.get("expected_version_label"),
                expect_conflict=bool(item.get("expect_conflict", False)),
                expected_citation_labels=item.get("expected_citation_labels", []),
            )
            for item in items
        ]
    except Exception as exc:
        st.error(
            f"Retrieval cases need query, relevant_ids, and retrieved fields: {exc}"
        )
        return None


def _parse_ingestion_cases(text: str) -> list[IngestionEvalCase] | None:
    try:
        items = json.loads(text)
    except json.JSONDecodeError as exc:
        st.error(f"Ingestion cases JSON is invalid: {exc}")
        return None

    try:
        return [
            IngestionEvalCase(
                document_id=item["document_id"],
                parse_success=bool(item.get("parse_success", False)),
                chunk_count=int(item.get("chunk_count", 0)),
                vector_node_count=int(item.get("vector_node_count", 0)),
                required_metadata_fields=item.get("required_metadata_fields", []),
                metadata_by_chunk=item.get("metadata_by_chunk", []),
                indexing_success=bool(item.get("indexing_success", True)),
                error_message=item.get("error_message"),
            )
            for item in items
        ]
    except Exception as exc:
        st.error(
            "Ingestion cases need document_id, parse_success, chunk_count, "
            f"and vector_node_count fields: {exc}"
        )
        return None


def _format_dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _active_nodes_for_document(document_id: str, client_id: str | None) -> int:
    with SessionLocal() as db:
        query = db.query(func.count(VectorNodeRegistry.id)).filter(
            VectorNodeRegistry.document_id == document_id,
            VectorNodeRegistry.is_active == True,
        )
        if client_id:
            query = query.filter(VectorNodeRegistry.client_id == client_id)
        return query.scalar() or 0
