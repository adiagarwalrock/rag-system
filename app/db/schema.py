from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.core.config import settings
from app.db.base import Base

logger = logging.getLogger(__name__)


def ensure_runtime_schema(engine: Engine) -> None:
    """
    Ensure runtime tables and additive columns exist.

    The project does not use a formal migration framework, so we apply safe,
    additive upgrades for local/dev compatibility.
    """
    Base.metadata.create_all(bind=engine)
    _ensure_query_logs_user_id(engine)
    _ensure_query_logs_session_id(engine)
    _ensure_query_logs_reasoning_effort(engine)
    _ensure_query_logs_llm_model(engine)
    _ensure_chat_messages_reasoning(engine)
    _ensure_chat_messages_citations_json(engine)
    _ensure_clients_embedding_model(engine)
    _backfill_clients_embedding_model(engine)
    _drop_legacy_auth_tables(engine)


def _ensure_clients_embedding_model(engine: Engine) -> None:
    _ensure_text_column(engine, table_name="clients", column_name="embedding_model")


def _backfill_clients_embedding_model(engine: Engine) -> None:
    """Set embedding_model to the configured default for any NULL client rows."""
    default = settings.EMBEDDING_MODEL
    try:
        inspector = inspect(engine)
        if "clients" not in set(inspector.get_table_names()):
            return
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    "UPDATE clients SET embedding_model = :m WHERE embedding_model IS NULL"
                ),
                {"m": default},
            )
        rows = getattr(result, "rowcount", 0) or 0
        if rows:
            logger.info(
                "Backfilled clients.embedding_model = '%s' for %d row(s).",
                default,
                rows,
            )
    except Exception:
        logger.exception("Failed to backfill clients.embedding_model")


def _ensure_query_logs_llm_model(engine: Engine) -> None:
    _ensure_text_column(engine, table_name="query_logs", column_name="llm_model")


def _ensure_chat_messages_reasoning(engine: Engine) -> None:
    _ensure_text_column(engine, table_name="chat_messages", column_name="reasoning")


def _ensure_chat_messages_citations_json(engine: Engine) -> None:
    _ensure_text_column(
        engine, table_name="chat_messages", column_name="citations_json"
    )


def _ensure_query_logs_session_id(engine: Engine) -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "query_logs" not in table_names:
        return

    columns = {column["name"] for column in inspector.get_columns("query_logs")}
    if "session_id" in columns:
        return

    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE query_logs ADD COLUMN session_id VARCHAR"))
        logger.info("Added query_logs.session_id runtime column.")
    except Exception:
        logger.exception("Failed to add query_logs.session_id runtime column")


def _ensure_query_logs_reasoning_effort(engine: Engine) -> None:
    _ensure_text_column(engine, table_name="query_logs", column_name="reasoning_effort")


def _ensure_query_logs_user_id(engine: Engine) -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "query_logs" not in table_names:
        return

    columns = {column["name"] for column in inspector.get_columns("query_logs")}
    if "user_id" in columns:
        return

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE query_logs ADD COLUMN user_id VARCHAR DEFAULT 'internal'"
                )
            )
        logger.info("Added query_logs.user_id runtime column.")
    except Exception:
        logger.exception("Failed to add query_logs.user_id runtime column")


def _ensure_text_column(engine: Engine, *, table_name: str, column_name: str) -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if table_name not in table_names:
        return

    columns = {column["name"] for column in inspector.get_columns(table_name)}
    if column_name in columns:
        return

    try:
        with engine.begin() as conn:
            conn.execute(
                text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} VARCHAR")
            )
        logger.info("Added %s.%s runtime column.", table_name, column_name)
    except Exception:
        logger.exception("Failed to add %s.%s runtime column", table_name, column_name)


def _drop_legacy_auth_tables(engine: Engine) -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    legacy_table_names = ("user_client_access", "user_roles", "roles", "users")

    for table_name in legacy_table_names:
        if table_name not in table_names:
            continue
        try:
            with engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            logger.info("Dropped legacy auth table: %s", table_name)
        except Exception:
            logger.exception("Failed to drop legacy auth table: %s", table_name)
