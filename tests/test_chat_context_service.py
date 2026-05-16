from datetime import datetime, timezone

from app.db.models import ChatMessage, ChatSession
from app.indexing.chat_history_store import ChatHistoryMatch
from app.services import chat_context_service
from app.services.chat_context_service import ChatContextService


def test_build_context_bundle_includes_recent_and_cross_session_pairs(
    db_session, seeded_entities, monkeypatch
):
    client = seeded_entities["client"]
    now = datetime.now(timezone.utc)
    session = ChatSession(
        id="session-current",
        client_id=client.id,
        title="Current",
        summary_text="Known priorities",
        created_at=now,
        updated_at=now,
        last_activity_at=now,
    )
    db_session.add(session)
    db_session.add_all(
        [
            ChatMessage(
                id="m1",
                client_id=client.id,
                session_id=session.id,
                role="user",
                content="First question",
                turn_index=1,
            ),
            ChatMessage(
                id="m2",
                client_id=client.id,
                session_id=session.id,
                role="assistant",
                content="First answer",
                turn_index=2,
            ),
            ChatMessage(
                id="m3",
                client_id=client.id,
                session_id=session.id,
                role="user",
                content="Current question",
                turn_index=3,
            ),
        ]
    )
    db_session.commit()

    monkeypatch.setattr(
        chat_context_service.chat_history_store,
        "search",
        lambda **kwargs: [
            ChatHistoryMatch(
                score=0.92,
                client_id=client.id,
                session_id="session-other",
                user_text="How do renewals work?",
                assistant_text="Renewals are annual with 30 day notice.",
                assistant_message_id="a-1",
                created_at=now.isoformat(),
            )
        ],
    )

    bundle = ChatContextService(db_session).build_context_bundle(
        client_id=client.id,
        session_id=session.id,
        current_question="Current question",
    )

    assert bundle.session_summary == "Known priorities"
    assert len(bundle.recent_turns) == 2
    assert bundle.recent_turns[0]["content"] == "First question"
    assert bundle.recent_turns[1]["content"] == "First answer"
    assert len(bundle.cross_session_pairs) == 1
    assert bundle.cross_session_pairs[0]["session_id"] == "session-other"
    assert bundle.cross_session_pairs[0]["score"] == 0.92
