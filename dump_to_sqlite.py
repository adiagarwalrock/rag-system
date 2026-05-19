"""
Dump all Snowflake tables into a local SQLite database (sf_dump.db).

Usage:
    python dump_to_sqlite.py

Requires SNOWFLAKE_* vars to be set in .env (read via app.core.config).
"""

import logging
import sys

import snowflake.connector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.logging_config import configure_logging
from app.db.base import Base

# Import models so Base.metadata is fully populated
from app.db.models.chat import ChatMessage, ChatSession  # noqa: F401
from app.db.models.client import Client  # noqa: F401
from app.db.models.document import (  # noqa: F401
    ConflictLog,
    Document,
    DocumentVersion,
    IngestionJob,
    QueryLog,
    RetrievalLog,
    VectorNodeRegistry,
)

configure_logging()
logger = logging.getLogger(__name__)

TABLES_IN_INSERT_ORDER = [
    "clients",
    "documents",
    "document_versions",
    "ingestion_jobs",
    "vector_node_registry",
    "chat_sessions",
    "query_logs",
    "retrieval_logs",
    "conflict_logs",
    "chat_messages",
]


def build_snowflake_connection():
    missing = [
        f
        for f in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_DATABASE",
                  "SNOWFLAKE_SCHEMA", "SNOWFLAKE_WAREHOUSE")
        if not getattr(settings, f, None)
    ]
    if missing:
        logger.error("Missing Snowflake settings: %s", missing)
        sys.exit(1)

    base_kwargs = dict(
        account=settings.SNOWFLAKE_ACCOUNT,
        user=settings.SNOWFLAKE_USER,
        database=settings.SNOWFLAKE_DATABASE,
        schema=settings.SNOWFLAKE_SCHEMA,
        warehouse=settings.SNOWFLAKE_WAREHOUSE,
        role=settings.SNOWFLAKE_ROLE,
    )

    if settings.SNOWFLAKE_TOKEN:
        logger.info("Using PAT authentication")
        return snowflake.connector.connect(
            **base_kwargs,
            token=settings.SNOWFLAKE_TOKEN,
            authenticator="programmatic_access_token",
        )

    if not settings.SNOWFLAKE_PASSWORD:
        logger.error("No SNOWFLAKE_PASSWORD or SNOWFLAKE_TOKEN set")
        sys.exit(1)

    logger.info("Using password authentication")
    return snowflake.connector.connect(**base_kwargs, password=settings.SNOWFLAKE_PASSWORD)


def main():
    logger.info("Connecting to Snowflake…")
    try:
        sf_conn = build_snowflake_connection()
        sf_conn.cursor().execute("SELECT 1")
        logger.info("Snowflake connection OK")
    except Exception as exc:
        logger.error("Cannot connect to Snowflake: %s", exc)
        sys.exit(1)

    logger.info("Creating sf_dump.db (SQLite)…")
    sqlite_engine = create_engine(
        "sqlite:///./sf_dump.db",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=sqlite_engine)
    SqliteSession = sessionmaker(bind=sqlite_engine, autocommit=False, autoflush=False)

    total_rows = 0
    with SqliteSession() as sqlite_sess:
        for table_name in TABLES_IN_INSERT_ORDER:
            try:
                cur = sf_conn.cursor(snowflake.connector.DictCursor)
                cur.execute(f'SELECT * FROM "{table_name.upper()}"')
                rows = cur.fetchall()
            except Exception as exc:
                logger.warning("Skipping %s — %s", table_name, exc)
                continue

            if not rows:
                logger.info("  %-30s  0 rows (empty)", table_name)
                continue

            # Snowflake returns uppercase column names; normalise to lowercase
            rows = [{k.lower(): v for k, v in row.items()} for row in rows]

            table = Base.metadata.tables[table_name]
            try:
                sqlite_sess.execute(table.insert(), rows)
                sqlite_sess.commit()
                logger.info("  %-30s  %d rows", table_name, len(rows))
                total_rows += len(rows)
            except Exception as exc:
                sqlite_sess.rollback()
                logger.error("Failed to insert into %s: %s", table_name, exc)

    sf_conn.close()
    logger.info("Done — %d total rows written to sf_dump.db", total_rows)


if __name__ == "__main__":
    main()
