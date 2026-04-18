import json
from datetime import datetime, timezone

from app.db.models import ChatMessage, ChatSession
from ui.lib import api as ui_api


def test_list_chat_messages_hydrates_assistant_result(db_session, seeded_entities, monkeypatch):
    client = seeded_entities["client"]
    now = datetime.now(timezone.utc)

    session = ChatSession(
        id="session-ui-1",
        client_id=client.id,
        title="Session",
        summary_text="",
        created_at=now,
        updated_at=now,
        last_activity_at=now,
    )
    db_session.add(session)
    db_session.add_all(
        [
            ChatMessage(
                id="msg-ui-1",
                client_id=client.id,
                session_id=session.id,
                role="user",
                content="What changed?",
                turn_index=1,
            ),
            ChatMessage(
                id="msg-ui-2",
                client_id=client.id,
                session_id=session.id,
                role="assistant",
                content="Policy v2 changed retention clauses.",
                reasoning="Retention changed from 30 to 45 days.",
                citations_json=json.dumps(
                    [
                        {
                            "citation_label": "policy_v2.pdf - p.3 - chunk 1",
                            "text": "Retention period is 45 days.",
                        }
                    ]
                ),
                turn_index=2,
                query_log_id="query-1",
            ),
        ]
    )
    db_session.commit()

    monkeypatch.setattr(ui_api, "SessionLocal", lambda: db_session)
    core = object.__new__(ui_api.VecteraCore)

    messages = core.list_chat_messages(session.id, limit=10)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert "result" not in messages[0]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["result"]["reasoning"] == "Retention changed from 30 to 45 days."
    assert messages[1]["result"]["query_id"] == "query-1"
    assert messages[1]["result"]["citations"] == [
        {
            "citation_label": "policy_v2.pdf - p.3 - chunk 1",
            "text": "Retention period is 45 days.",
        }
    ]


def test_list_chat_messages_handles_invalid_citations_json(
    db_session, seeded_entities, monkeypatch
):
    client = seeded_entities["client"]
    now = datetime.now(timezone.utc)

    session = ChatSession(
        id="session-ui-2",
        client_id=client.id,
        title="Session",
        summary_text="",
        created_at=now,
        updated_at=now,
        last_activity_at=now,
    )
    db_session.add(session)
    db_session.add(
        ChatMessage(
            id="msg-ui-invalid",
            client_id=client.id,
            session_id=session.id,
            role="assistant",
            content="Answer",
            reasoning="Trace",
            citations_json="{not-valid-json",
            turn_index=1,
            query_log_id="query-2",
        )
    )
    db_session.commit()

    monkeypatch.setattr(ui_api, "SessionLocal", lambda: db_session)
    core = object.__new__(ui_api.VecteraCore)

    messages = core.list_chat_messages(session.id, limit=10)

    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["result"]["reasoning"] == "Trace"
    assert messages[0]["result"]["query_id"] == "query-2"
    assert messages[0]["result"]["citations"] == []
