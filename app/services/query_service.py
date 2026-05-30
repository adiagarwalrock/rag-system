"""
Query service: orchestrates retrieval, answer generation, and logging.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import insert, true
from sqlalchemy.orm import Session

from app.db.models.document import (
    ConflictLog,
    QueryLog,
    RetrievalLog,
    VectorNodeRegistry,
)
from app.retrieval.retriever import VecteraRetriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueryExecutionRequest:
    question: str
    client_id: str
    reasoning_effort: str = "medium"
    reasoning_summary: str | None = None
    session_id: str | None = None
    conversation_context: dict[str, Any] | None = None
    status_callback: Callable[[str], None] | None = field(default=None, compare=False)
    reasoning_callback: Callable[[str], None] | None = field(default=None, compare=False)
    answer_callback: Callable[[str], None] | None = field(default=None, compare=False)


class QueryLogWriter:
    """Persist query lifecycle, retrieval logs, and conflict logs."""

    def __init__(self, db: Session):
        self.db = db

    def create_running_log(self, request: QueryExecutionRequest) -> QueryLog:
        query_log = QueryLog(
            id=str(uuid.uuid4()),
            client_id=request.client_id,
            user_id="internal",
            session_id=request.session_id,
            question=request.question,
            status="running",
        )
        self.db.add(query_log)
        self.db.commit()
        return query_log

    def build_registry_by_node_id(
        self,
        *,
        client_id: str,
        citations: list[dict[str, Any]],
    ) -> dict[str, VectorNodeRegistry]:
        vector_node_ids = [
            citation.get("vector_node_id")
            for citation in citations
            if citation.get("vector_node_id")
        ]
        if not vector_node_ids:
            return {}

        rows = (
            self.db.query(VectorNodeRegistry)
            .filter(
                VectorNodeRegistry.client_id == client_id,
                VectorNodeRegistry.vector_node_id.in_(vector_node_ids),
                VectorNodeRegistry.is_active == true(),
            )
            .all()
        )
        return {row.vector_node_id: row for row in rows}

    def persist_retrieval_logs(
        self,
        *,
        query_log_id: str,
        citations: list[dict[str, Any]],
        registry_by_node_id: dict[str, VectorNodeRegistry],
    ) -> None:
        for index, citation in enumerate(citations, start=1):
            vector_node_id = citation.get("vector_node_id")
            registry_row = registry_by_node_id.get(vector_node_id)
            values = {
                "id": str(uuid.uuid4()),
                "query_log_id": query_log_id,
                "vector_node_id": vector_node_id,
                "document_id": citation.get("document_id")
                or (registry_row.document_id if registry_row else None),
                "rank": index,
                "retrieval_score": citation.get("score"),
            }
            self.db.execute(insert(RetrievalLog).values(**values))

    def persist_conflict_logs(
        self,
        *,
        query_log_id: str,
        conflicts: list[dict[str, Any]],
    ) -> None:
        for conflict in conflicts:
            values = {
                "id": str(uuid.uuid4()),
                "query_log_id": query_log_id,
                "conflict_type": conflict.get("conflict_type", "unknown"),
                "summary": conflict.get("summary"),
            }
            self.db.execute(insert(ConflictLog).values(**values))

    @staticmethod
    def mark_success(*, query_log: QueryLog, answer: str, latency_ms: int) -> None:
        query_log.answer = answer[:5000]
        query_log.status = "completed"
        query_log.latency_ms = latency_ms

    @staticmethod
    def mark_failed(*, query_log: QueryLog, error: Exception, latency_ms: int) -> None:
        query_log.status = "failed"
        query_log.answer = f"Error: {str(error)[:500]}"
        query_log.latency_ms = latency_ms


class QueryExecutionService:
    """Runs retrieval and coordinates persistence for one query."""

    def __init__(self, db: Session, retriever_factory: Any | None = None):
        self.db = db
        self.retriever_factory = retriever_factory or VecteraRetriever
        self.log_writer = QueryLogWriter(db)

    def execute(self, request: QueryExecutionRequest) -> dict[str, Any]:
        start_time = time.time()
        query_log = self.log_writer.create_running_log(request)

        try:
            result = self._run_retrieval(request)
            latency_ms = self._latency_ms(start_time)

            citations = result.get("citations", [])
            registry_by_node_id = self.log_writer.build_registry_by_node_id(
                client_id=request.client_id,
                citations=citations,
            )
            self.log_writer.persist_retrieval_logs(
                query_log_id=query_log.id,
                citations=citations,
                registry_by_node_id=registry_by_node_id,
            )
            self.log_writer.persist_conflict_logs(
                query_log_id=query_log.id,
                conflicts=result.get("conflicts", []),
            )

            self.log_writer.mark_success(
                query_log=query_log,
                answer=result["answer"],
                latency_ms=latency_ms,
            )
            self.db.commit()
            result.update({"query_id": query_log.id, "latency_ms": latency_ms})
            return result

        except Exception as exc:
            logger.error("Query failed: %s", str(exc))
            self.log_writer.mark_failed(
                query_log=query_log,
                error=exc,
                latency_ms=self._latency_ms(start_time),
            )
            self.db.commit()
            raise

    def _run_retrieval(self, request: QueryExecutionRequest) -> dict[str, Any]:
        retriever = self.retriever_factory(
            client_id=request.client_id,
            reasoning_effort=request.reasoning_effort,
            reasoning_summary=request.reasoning_summary,
            conversation_context=request.conversation_context,
            reasoning_callback=request.reasoning_callback,
            answer_callback=request.answer_callback,
        )
        return retriever.query(request.question, status_callback=request.status_callback)

    @staticmethod
    def _latency_ms(start_time: float) -> int:
        return int((time.time() - start_time) * 1000)


def execute_query(
    question: str,
    client_id: str,
    db: Session,
    reasoning_effort: str = "medium",
    reasoning_summary: str | None = None,
    session_id: str | None = None,
    conversation_context: dict | None = None,
    status_callback: Callable[[str], None] | None = None,
    reasoning_callback: Callable[[str], None] | None = None,
    answer_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Execute a full query pipeline: retrieve, answer, log.

    ``status_callback`` is called at each pipeline stage with a human-readable label.
    Callers (e.g. the Streamlit UI) can use this to update a live progress panel while
    the synchronous pipeline runs.

    Returns:
        Dict with answer, citations, conflicts, and query metadata.
    """
    from app.core.config import settings

    request = QueryExecutionRequest(
        question=question,
        client_id=client_id,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        session_id=session_id,
        conversation_context=conversation_context,
        status_callback=status_callback,
        reasoning_callback=reasoning_callback,
        answer_callback=answer_callback,
    )

    if settings.ENABLE_AGENTIC_RAG:
        from app.agents.adapter import AgenticRetrieverAdapter
        return QueryExecutionService(db, retriever_factory=AgenticRetrieverAdapter).execute(request)

    return QueryExecutionService(db).execute(request)
