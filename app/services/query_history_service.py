"""
Query history service: list persisted query logs with summary aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Query, Session

from app.db.models.document import ConflictLog, QueryLog, RetrievalLog


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
                    QueryLog.user_id.label("user_id"),
                    QueryLog.client_id.label("client_id"),
                    QueryLog.question.label("question"),
                    QueryLog.answer.label("answer"),
                    QueryLog.status.label("status"),
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
                QueryLog.user_id,
                QueryLog.client_id,
                QueryLog.question,
                QueryLog.answer,
                QueryLog.status,
                QueryLog.latency_ms,
            )
            .order_by(QueryLog.created_at.desc())
            .offset(normalized.offset)
            .limit(normalized.limit)
            .all()
        )

        return {
            "rows": [
                {
                    "query_id": row.query_id,
                    "created_at": row.created_at,
                    "user_id": row.user_id,
                    "client_id": row.client_id,
                    "question": row.question,
                    "answer": row.answer,
                    "status": row.status,
                    "latency_ms": row.latency_ms,
                    "retrieval_count": int(row.retrieval_count or 0),
                    "conflict_count": int(row.conflict_count or 0),
                }
                for row in rows
            ],
            "total": total,
        }

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
