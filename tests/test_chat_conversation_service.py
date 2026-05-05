import json
from datetime import datetime, timezone

from app.db.models import ChatMessage, ChatSession
from app.services import chat_conversation_service
from app.services.chat_context_service import ChatContextBundle
from app.services.chat_conversation_service import ChatConversationService


def test_execute_client_query_creates_session_and_persists_turns(
    db_session, seeded_entities, monkeypatch
):
    client = seeded_entities["client"]
    service = ChatConversationService(db_session)

    captured: dict = {}
    indexed: dict = {}

    def _fake_execute_query(
        question: str,
        client_id: str,
        db,
        reasoning_effort: str = "medium",
        session_id: str | None = None,
        conversation_context: dict | None = None,
    ) -> dict:
        captured["question"] = question
        captured["client_id"] = client_id
        captured["reasoning_effort"] = reasoning_effort
        captured["session_id"] = session_id
        captured["conversation_context"] = conversation_context
        return {
            "answer": "Policy v2 changed retention clauses.",
            "reasoning": "Policy v2 updated retention from 30 to 45 days.",
            "citations": [
                {
                    "citation_label": "policy_v2.pdf - p.3 - chunk 1",
                    "text": "Retention period is 45 days.",
                    "vector_node_id": "node-1",
                }
            ],
            "conflicts": [],
            "query_id": "query-123",
            "latency_ms": 123,
            "reasoning_effort": reasoning_effort,
            "reasoning_effort_applied": True,
        }

    monkeypatch.setattr(chat_conversation_service, "execute_query", _fake_execute_query)
    monkeypatch.setattr(
        service.context_service,
        "build_context_bundle",
        lambda **kwargs: ChatContextBundle(
            session_summary="Summary",
            recent_turns=[{"role": "user", "content": "Earlier context"}],
            cross_session_pairs=[],
        ),
    )
    monkeypatch.setattr(
        service.context_service,
        "index_qa_pair",
        lambda **kwargs: indexed.update(kwargs),
    )

    monkeypatch.setattr(
        chat_conversation_service,
        "invoke_llm_chat",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        chat_conversation_service,
        "extract_chat_response_text",
        lambda response: "Updated session summary",
    )

    result = service.execute_client_query(
        client_id=client.id,
        question="What changed in policy v2?",
        reasoning_effort="high",
    )

    assert result["session_id"]
    assert captured["session_id"] == result["session_id"]
    assert captured["client_id"] == client.id
    assert captured["reasoning_effort"] == "high"
    assert captured["conversation_context"]["session_summary"] == "Summary"

    sessions = (
        db_session.query(ChatSession).filter(ChatSession.client_id == client.id).all()
    )
    assert len(sessions) == 1
    assert sessions[0].summary_text == "Updated session summary"
    assert sessions[0].title == "What changed in policy v2?"

    messages = (
        db_session.query(ChatMessage)
        .filter(ChatMessage.session_id == result["session_id"])
        .order_by(ChatMessage.turn_index.asc())
        .all()
    )
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "What changed in policy v2?"
    assert messages[1].role == "assistant"
    assert messages[1].query_log_id == "query-123"
    assert messages[1].reasoning == "Policy v2 updated retention from 30 to 45 days."
    assert json.loads(messages[1].citations_json or "[]") == [
        {
            "citation_label": "policy_v2.pdf - p.3 - chunk 1",
            "text": "Retention period is 45 days.",
            "vector_node_id": "node-1",
        }
    ]

    assert indexed["client_id"] == client.id
    assert indexed["session_id"] == result["session_id"]
    assert indexed["assistant_message_id"] == messages[1].id


def test_clear_session_removes_messages_and_resets_summary(
    db_session, seeded_entities, monkeypatch
):
    client = seeded_entities["client"]
    service = ChatConversationService(db_session)
    now = datetime.now(timezone.utc)

    session = ChatSession(
        id="session-1",
        client_id=client.id,
        title="Context",
        summary_text="Old summary",
        created_at=now,
        updated_at=now,
        last_activity_at=now,
    )
    db_session.add(session)
    db_session.add_all(
        [
            ChatMessage(
                id="msg-1",
                client_id=client.id,
                session_id=session.id,
                role="user",
                content="Question",
                turn_index=1,
            ),
            ChatMessage(
                id="msg-2",
                client_id=client.id,
                session_id=session.id,
                role="assistant",
                content="Answer",
                turn_index=2,
            ),
        ]
    )
    db_session.commit()

    deleted = {"called": False}
    monkeypatch.setattr(
        service.context_service,
        "delete_session_memory",
        lambda session_id: deleted.__setitem__("called", True),
    )

    service.clear_session(session_id=session.id)

    remaining = (
        db_session.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .count()
    )
    assert remaining == 0
    refreshed = db_session.query(ChatSession).filter(ChatSession.id == session.id).one()
    assert refreshed.summary_text == ""
    assert deleted["called"] is True
