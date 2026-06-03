from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    client_id = Column(String, ForeignKey("clients.id"), nullable=False)
    name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    checksum = Column(String, nullable=True)
    status = Column(String, nullable=False, default="uploaded")
    source_label = Column(String, nullable=True)
    document_family = Column(String, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    client = relationship("Client", back_populates="documents")
    versions = relationship("DocumentVersion", back_populates="document")
    ingestion_jobs = relationship("IngestionJob", back_populates="document")


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    version_label = Column(String, nullable=True)
    version_group = Column(String, nullable=True)
    version_rank = Column(Integer, nullable=True)
    published_at = Column(DateTime, nullable=True)
    last_modified_at = Column(DateTime, nullable=True)
    effective_from = Column(DateTime, nullable=True)
    effective_to = Column(DateTime, nullable=True)
    is_current = Column(Boolean, default=False)
    confidence_score = Column(Float, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    document = relationship("Document", back_populates="versions")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(String, primary_key=True)
    client_id = Column(String, ForeignKey("clients.id"), nullable=False)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    status = Column(String, nullable=False, default="queued")
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    parser_name = Column(String, nullable=True)
    parser_version = Column(String, nullable=True)
    filesize_bytes = Column(Integer, nullable=True)

    document = relationship("Document", back_populates="ingestion_jobs")


class VectorNodeRegistry(Base):
    __tablename__ = "vector_node_registry"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    client_id = Column(String, ForeignKey("clients.id"), nullable=False)
    vector_collection = Column(String, nullable=False)
    vector_node_id = Column(String, nullable=False)
    embedding_model = Column(String, nullable=True)
    indexed_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    is_active = Column(Boolean, default=True)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(String, primary_key=True)
    client_id = Column(String, ForeignKey("clients.id"), nullable=False)
    # Legacy Snowflake deployments may still require USER_ID to be non-null.
    # In single-tenant mode we keep a system sentinel.
    user_id = Column(String, nullable=True, default="internal")
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=True)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=True)
    status = Column(String, nullable=False, default="completed")
    llm_model = Column(String, nullable=True)
    reasoning_effort = Column(String, nullable=True, default="medium")
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    latency_ms = Column(Integer, nullable=True)

    retrieval_logs = relationship("RetrievalLog", back_populates="query_log")
    conflict_logs = relationship("ConflictLog", back_populates="query_log")
    chat_messages = relationship("ChatMessage", back_populates="query_log")
    session = relationship("ChatSession", back_populates="query_logs")


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id = Column(String, primary_key=True)
    query_log_id = Column(String, ForeignKey("query_logs.id"), nullable=False)
    vector_node_id = Column(String, nullable=True)
    document_id = Column(String, nullable=True)
    rank = Column(Integer, nullable=False)
    retrieval_score = Column(Float, nullable=True)
    rerank_score = Column(Float, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    query_log = relationship("QueryLog", back_populates="retrieval_logs")


class ConflictLog(Base):
    __tablename__ = "conflict_logs"

    id = Column(String, primary_key=True)
    query_log_id = Column(String, ForeignKey("query_logs.id"), nullable=False)
    conflict_type = Column(String, nullable=False)
    summary = Column(String, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    query_log = relationship("QueryLog", back_populates="conflict_logs")
