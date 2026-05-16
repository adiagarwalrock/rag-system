from datetime import datetime, timedelta, timezone

from app.db.models import Client, ConflictLog, QueryLog, RetrievalLog
from app.services.query_history_service import QueryHistoryFilters, QueryHistoryService


def _seed_history_rows(db_session):
    client_two = Client(
        id="client-2",
        name="Beta Co",
        description="Second client",
        is_active=True,
    )
    db_session.add(client_two)

    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    q1 = QueryLog(
        id="query-1",
        client_id="client-1",
        question="What is the renewal deadline?",
        answer="Renewal is due in 30 days.",
        status="completed",
        created_at=base_time,
        latency_ms=1450,
    )
    q2 = QueryLog(
        id="query-2",
        client_id="client-1",
        question="Show policy v2 changes",
        answer="Error: upstream timeout",
        status="failed",
        created_at=base_time + timedelta(minutes=1),
        latency_ms=2750,
    )
    q3 = QueryLog(
        id="query-3",
        client_id="client-2",
        question="Summarize latest terms",
        answer=None,
        status="running",
        created_at=base_time + timedelta(minutes=2),
        latency_ms=None,
    )
    db_session.add_all([q1, q2, q3])

    db_session.add_all(
        [
            RetrievalLog(
                id="retrieval-1",
                query_log_id="query-1",
                vector_node_id="node-1",
                document_id="doc-1",
                rank=1,
                retrieval_score=0.91,
            ),
            RetrievalLog(
                id="retrieval-2",
                query_log_id="query-1",
                vector_node_id="node-2",
                document_id="doc-1",
                rank=2,
                retrieval_score=0.75,
            ),
            RetrievalLog(
                id="retrieval-3",
                query_log_id="query-2",
                vector_node_id="node-3",
                document_id="doc-1",
                rank=1,
                retrieval_score=0.68,
            ),
            ConflictLog(
                id="conflict-1",
                query_log_id="query-2",
                conflict_type="numeric_conflict",
                summary="Two sources disagree on retention period.",
            ),
            ConflictLog(
                id="conflict-2",
                query_log_id="query-2",
                conflict_type="policy_conflict",
                summary="Policy wording differs across versions.",
            ),
        ]
    )

    db_session.commit()


def test_list_query_history_returns_newest_first_with_aggregates(
    db_session, seeded_entities
):
    _seed_history_rows(db_session)
    service = QueryHistoryService(db_session)

    result = service.list_query_history(QueryHistoryFilters(limit=10, offset=0))

    assert result["total"] == 3
    assert [row["query_id"] for row in result["rows"]] == [
        "query-3",
        "query-2",
        "query-1",
    ]

    row_by_id = {row["query_id"]: row for row in result["rows"]}
    assert row_by_id["query-1"]["retrieval_count"] == 2
    assert row_by_id["query-1"]["conflict_count"] == 0
    assert row_by_id["query-2"]["retrieval_count"] == 1
    assert row_by_id["query-2"]["conflict_count"] == 2
    assert row_by_id["query-3"]["retrieval_count"] == 0
    assert row_by_id["query-3"]["conflict_count"] == 0


def test_list_query_history_filters_by_client_status_and_question_search(
    db_session, seeded_entities
):
    _seed_history_rows(db_session)
    service = QueryHistoryService(db_session)

    client_filtered = service.list_query_history(
        QueryHistoryFilters(client_id="client-1", limit=10, offset=0)
    )
    assert client_filtered["total"] == 2
    assert {row["query_id"] for row in client_filtered["rows"]} == {
        "query-1",
        "query-2",
    }

    status_filtered = service.list_query_history(
        QueryHistoryFilters(status="failed", limit=10, offset=0)
    )
    assert status_filtered["total"] == 1
    assert status_filtered["rows"][0]["query_id"] == "query-2"

    search_filtered = service.list_query_history(
        QueryHistoryFilters(search_text="ReNeWaL", limit=10, offset=0)
    )
    assert search_filtered["total"] == 1
    assert search_filtered["rows"][0]["query_id"] == "query-1"


def test_list_query_history_supports_pagination(db_session, seeded_entities):
    _seed_history_rows(db_session)
    service = QueryHistoryService(db_session)

    result = service.list_query_history(QueryHistoryFilters(limit=1, offset=1))

    assert result["total"] == 3
    assert len(result["rows"]) == 1
    assert result["rows"][0]["query_id"] == "query-2"
