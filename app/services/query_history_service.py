"""
Query history service: list persisted query logs with summary aggregates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Query, Session

from app.db.models.chat import ChatMessage
from app.db.models.document import ConflictLog, Document, QueryLog, RetrievalLog


@dataclass(frozen=True)
class QueryHistoryFilters:
    client_id: str | None = None
    status: str | None = None
    search_text: str | None = None
    limit: int = 50
    offset: int = 0


class QueryHistoryService:
    """Read-only query history workflows."""

    def __init__(self, db: Session):
        self.db = db

    def list_query_history(self, filters: QueryHistoryFilters) -> dict[str, Any]:
        normalized = QueryHistoryFilters(
            client_id=filters.client_id,
            status=self._normalize_status(filters.status),
            search_text=(filters.search_text or "").strip(),
            limit=max(1, min(filters.limit, 200)),
            offset=max(0, filters.offset),
        )

        total = self._apply_filters(self.db.query(QueryLog), normalized).count()

        rows = (
            self._apply_filters(
                self.db.query(
                    QueryLog.id.label("query_id"),
                    QueryLog.created_at.label("created_at"),
                    QueryLog.client_id.label("client_id"),
                    QueryLog.session_id.label("session_id"),
                    QueryLog.question.label("question"),
                    QueryLog.answer.label("answer"),
                    QueryLog.status.label("status"),
                    QueryLog.reasoning_effort.label("reasoning_effort"),
                    QueryLog.latency_ms.label("latency_ms"),
                    func.count(func.distinct(RetrievalLog.id)).label("retrieval_count"),
                    func.count(func.distinct(ConflictLog.id)).label("conflict_count"),
                )
                .outerjoin(RetrievalLog, RetrievalLog.query_log_id == QueryLog.id)
                .outerjoin(ConflictLog, ConflictLog.query_log_id == QueryLog.id),
                normalized,
            )
            .group_by(
                QueryLog.id,
                QueryLog.created_at,
                QueryLog.client_id,
                QueryLog.session_id,
                QueryLog.question,
                QueryLog.answer,
                QueryLog.status,
                QueryLog.reasoning_effort,
                QueryLog.latency_ms,
            )
            .order_by(QueryLog.created_at.desc())
            .offset(normalized.offset)
            .limit(normalized.limit)
            .all()
        )
        query_ids = [row.query_id for row in rows]
        citations_by_query = self._citations_by_query_id(query_ids)
        retrieval_citations_by_query = self._retrieval_citations_by_query_id(query_ids)
        conflicts_by_query = self._conflicts_by_query_id(query_ids)

        return {
            "rows": [
                {
                    "query_id": row.query_id,
                    "created_at": row.created_at,
                    "client_id": row.client_id,
                    "session_id": row.session_id,
                    "question": row.question,
                    "answer": row.answer,
                    "status": row.status,
                    "reasoning_effort": row.reasoning_effort or "medium",
                    "latency_ms": row.latency_ms,
                    "retrieval_count": int(row.retrieval_count or 0),
                    "conflict_count": int(row.conflict_count or 0),
                    "citations": (
                        c
                        if (c := citations_by_query.get(row.query_id)) is not None
                        else retrieval_citations_by_query.get(row.query_id, [])
                    ),
                    "conflicts": conflicts_by_query.get(row.query_id, []),
                }
                for row in rows
            ],
            "total": total,
        }

    def delete_query_history_item(self, *, client_id: str, query_id: str) -> None:
        query_log = (
            self.db.query(QueryLog)
            .filter(QueryLog.id == query_id, QueryLog.client_id == client_id)
            .first()
        )
        if not query_log:
            raise ValueError(f"Query history item {query_id} not found.")

        self.db.query(RetrievalLog).filter(
            RetrievalLog.query_log_id == query_id
        ).delete(synchronize_session=False)
        self.db.query(ConflictLog).filter(
            ConflictLog.query_log_id == query_id
        ).delete(synchronize_session=False)
        self.db.query(ChatMessage).filter(ChatMessage.query_log_id == query_id).update(
            {ChatMessage.query_log_id: None},
            synchronize_session=False,
        )
        self.db.delete(query_log)
        self.db.commit()

    def _citations_by_query_id(self, query_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not query_ids:
            return {}

        rows = (
            self.db.query(ChatMessage.query_log_id, ChatMessage.citations_json)
            .filter(
                ChatMessage.query_log_id.in_(query_ids),
                ChatMessage.role == "assistant",
            )
            .all()
        )
        result: dict[str, list[dict[str, Any]]] = {}
        for query_id, citations_json in rows:
            citations = self._decode_citations(citations_json)
            if citations:
                result[query_id] = citations
        return result

    def _retrieval_citations_by_query_id(
        self, query_ids: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        if not query_ids:
            return {}

        rows = (
            self.db.query(RetrievalLog, Document.name)
            .outerjoin(Document, Document.id == RetrievalLog.document_id)
            .filter(RetrievalLog.query_log_id.in_(query_ids))
            .order_by(RetrievalLog.query_log_id, RetrievalLog.rank)
            .all()
        )
        result: dict[str, list[dict[str, Any]]] = {}
        for retrieval, document_name in rows:
            result.setdefault(retrieval.query_log_id, []).append(
                {
                    "document_id": retrieval.document_id,
                    "filename": document_name or retrieval.document_id or "Unknown source",
                    "chunk_id": retrieval.vector_node_id,
                    "rank": retrieval.rank,
                    "score": retrieval.retrieval_score,
                    "rerank_score": retrieval.rerank_score,
                }
            )
        return result

    def _conflicts_by_query_id(self, query_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not query_ids:
            return {}

        rows = (
            self.db.query(ConflictLog)
            .filter(ConflictLog.query_log_id.in_(query_ids))
            .order_by(ConflictLog.query_log_id, ConflictLog.created_at)
            .all()
        )
        result: dict[str, list[dict[str, Any]]] = {}
        for conflict in rows:
            result.setdefault(conflict.query_log_id, []).append(
                {
                    "type": conflict.conflict_type,
                    "conflict_type": conflict.conflict_type,
                    "severity": "medium",
                    "explanation": conflict.summary or "Conflict detected.",
                }
            )
        return result

    @staticmethod
    def _decode_citations(value: str | None) -> list[dict[str, Any]]:
        if not value:
            return []
        try:
            decoded = json.loads(value)
        except Exception:
            return []
        if not isinstance(decoded, list):
            return []
        return [item for item in decoded if isinstance(item, dict)]

    def _apply_filters(self, query: Query, filters: QueryHistoryFilters) -> Query:
        if filters.client_id:
            query = query.filter(QueryLog.client_id == filters.client_id)

        if filters.status:
            query = query.filter(QueryLog.status == filters.status)

        if filters.search_text:
            needle = f"%{filters.search_text.lower()}%"
            query = query.filter(func.lower(QueryLog.question).like(needle))

        return query

    @staticmethod
    def _normalize_status(status: str | None) -> str | None:
        if not status:
            return None
        cleaned = status.strip().lower()
        if cleaned in {"", "all"}:
            return None
        return cleaned
