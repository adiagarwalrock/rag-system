from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

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
                text("ALTER TABLE query_logs ADD COLUMN user_id VARCHAR DEFAULT 'internal'")
            )
        logger.info("Added query_logs.user_id runtime column.")
    except Exception:
        logger.exception("Failed to add query_logs.user_id runtime column")
