import logging
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy import event

from app.core.config import settings
from app.core.logging_config import configure_logging

configure_logging()

logger = logging.getLogger(__name__)


def get_engine():
    if settings.SNOWFLAKE_ACCOUNT and settings.SNOWFLAKE_USER:
        logger.info("Connecting to Snowflake database")
        # URL-encode the password in case it contains special characters
        password = quote_plus(settings.SNOWFLAKE_PASSWORD or "")
        conn_str = (
            f"snowflake://{settings.SNOWFLAKE_USER}:{password}"
            f"@{settings.SNOWFLAKE_ACCOUNT}"
            f"/{settings.SNOWFLAKE_DATABASE}"
            f"/{settings.SNOWFLAKE_SCHEMA}"
            f"?warehouse={settings.SNOWFLAKE_WAREHOUSE}"
            f"&role={settings.SNOWFLAKE_ROLE}"
        )
        return create_engine(conn_str, echo=False)
    # Fallback to local SQLite for rapid development and testing.
    # WAL mode + busy_timeout allow concurrent readers/writers from multiple threads
    # without "database is locked" deadlocks during parallel eval runs.
    logger.info("Connecting to local SQLite database fallback")

    _engine = create_engine(
        "sqlite:///./rag_local.db",
        connect_args={"check_same_thread": False, "timeout": 30},
        # Recycle stale pooled connections automatically — prevents "database
        # disk image is malformed" errors after an unclean shutdown left a
        # WAL connection in a bad in-memory state.
        pool_pre_ping=True,
    )

    @event.listens_for(_engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")  # 30s before giving up
        cursor.execute("PRAGMA synchronous=NORMAL")
        # Force a full WAL checkpoint on every new connection so stale WAL
        # frames from a previous crash are merged and discarded before any
        # query runs on this connection.
        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        cursor.close()

    return _engine


engine = get_engine()

# Validate the connection early so credential/network errors surface at startup
try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection validated successfully")
except Exception as e:
    logger.error("Database connection FAILED at startup: %s", e)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
