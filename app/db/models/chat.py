from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db.base import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True)
    client_id = Column(String, ForeignKey("clients.id"), nullable=False)
    title = Column(String, nullable=True)
    summary_text = Column(String, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    last_activity_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    client = relationship("Client", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session")
    query_logs = relationship("QueryLog", back_populates="session")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    client_id = Column(String, ForeignKey("clients.id"), nullable=False)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    reasoning = Column(Text, nullable=True)
    citations_json = Column(Text, nullable=True)
    turn_index = Column(Integer, nullable=False)
    query_log_id = Column(String, ForeignKey("query_logs.id"), nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    client = relationship("Client", back_populates="chat_messages")
    session = relationship("ChatSession", back_populates="messages")
    query_log = relationship("QueryLog", back_populates="chat_messages")
