"""
Query service: orchestrates retrieval, answer generation, and logging.
"""

import logging
import time
import uuid

from sqlalchemy import insert
from sqlalchemy.orm import Session

from app.db.models.document import (
    ConflictLog,
    QueryLog,
    RetrievalLog,
    VectorNodeRegistry,
)
from app.retrieval.retriever import VecteraRetriever

logger = logging.getLogger(__name__)


def execute_query(
    question: str,
    client_id: str,
    db: Session,
    reasoning_effort: str = "medium",
) -> dict:
    """
    Execute a full query pipeline: retrieve, answer, log.

    Returns:
        Dict with answer, citations, conflicts, and query metadata.
    """
    start_time = time.time()
    query_log_id = str(uuid.uuid4())

    # Create query log record
    query_log = QueryLog(
        id=query_log_id,
        client_id=client_id,
        question=question,
        status="running",
    )
    db.add(query_log)
    db.commit()

    try:
        # Execute retrieval pipeline
        retriever = VecteraRetriever(
            client_id=client_id,
            reasoning_effort=reasoning_effort,
        )
        result = retriever.query(question)

        latency_ms = int((time.time() - start_time) * 1000)

        # Update query log
        query_log.answer = result["answer"][:5000]
        query_log.status = "completed"
        query_log.latency_ms = latency_ms

        # Build point-id -> registry mapping for scalar retrieval references.
        citations = result.get("citations", [])
        vector_node_ids = [
            cit.get("vector_node_id") for cit in citations if cit.get("vector_node_id")
        ]
        registry_by_node_id = {}
        if vector_node_ids:
            rows = (
                db.query(VectorNodeRegistry)
                .filter(
                    VectorNodeRegistry.client_id == client_id,
                    VectorNodeRegistry.vector_node_id.in_(vector_node_ids),
                    VectorNodeRegistry.is_active == True,
                )
                .all()
            )
            registry_by_node_id = {row.vector_node_id: row for row in rows}

        # Log citations
        for i, cit in enumerate(citations):
            vector_node_id = cit.get("vector_node_id")
            registry_row = registry_by_node_id.get(vector_node_id)
            vals = {
                "id": str(uuid.uuid4()),
                "query_log_id": query_log_id,
                "vector_node_id": vector_node_id,
                "document_id": cit.get("document_id")
                or (registry_row.document_id if registry_row else None),
                "rank": i + 1,
                "retrieval_score": cit.get("score"),
            }
            db.execute(insert(RetrievalLog).values(**vals))

        # Log conflicts
        for conf in result.get("conflicts", []):
            vals = {
                "id": str(uuid.uuid4()),
                "query_log_id": query_log_id,
                "conflict_type": conf.get("conflict_type", "unknown"),
                "summary": conf.get("summary"),
            }
            db.execute(insert(ConflictLog).values(**vals))

        db.commit()
        result.update({"query_id": query_log_id, "latency_ms": latency_ms})
        return result

    except Exception as e:
        logger.error("Query failed: %s", str(e))
        query_log.status = "failed"
        query_log.answer = f"Error: {str(e)[:500]}"
        query_log.latency_ms = int((time.time() - start_time) * 1000)
        db.commit()
        raise
