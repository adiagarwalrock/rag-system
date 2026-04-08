import logging

from app.db.base import Base
from app.db.models.client import Client
from app.db.models.document import (
    Document,
    DocumentVersion,
    IngestionJob,
    VectorNodeRegistry,
)

# Import all models to ensure they are registered with Base.metadata
from app.db.models.user import Role, User, UserClientAccess, UserRole
from app.db.snowflake import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_db():
    logger.info("Initializing database schema...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized successfully (Snowflake/SQLite).")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")


if __name__ == "__main__":
    init_db()
