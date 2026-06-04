"""Backfill clients.embedding_model and VectorNodeRegistry.embedding_model for NULL rows.

Sets the value to the current settings.EMBEDDING_MODEL default for any row that was
created before the per-client embedding model feature was introduced.

Run:
    uv run python -m app.scripts.backfill_clients_embedding_model
"""

import logging

from sqlalchemy import update

from app.core.config import settings
from app.core.logging_config import configure_logging
from app.db.models.client import Client
from app.db.models.document import VectorNodeRegistry
from app.db.snowflake import SessionLocal

logger = logging.getLogger(__name__)


def _backfill_model_field(model_cls, default_model: str) -> tuple[int, int]:
    """Bulk-set embedding_model = default_model for all NULL rows of model_cls.

    Returns (updated, already_set).
    """
    with SessionLocal() as db:
        total: int = db.query(model_cls).count()
        updated: int = (
            db.execute(
                update(model_cls)
                .where(model_cls.embedding_model == None)  # noqa: E711
                .values(embedding_model=default_model)
            )
        ).rowcount or 0
        db.commit()
    return updated, total - updated


def backfill_clients(default_model: str) -> tuple[int, int]:
    """Set embedding_model on every client that currently has NULL.

    Returns (updated, already_set).
    """
    return _backfill_model_field(Client, default_model)


def backfill_vector_registry(default_model: str) -> tuple[int, int]:
    """Set embedding_model on every VectorNodeRegistry row that currently has NULL.

    Returns (updated, already_set).
    """
    return _backfill_model_field(VectorNodeRegistry, default_model)


def main() -> None:
    configure_logging()
    logger.setLevel(logging.INFO)

    default_model = settings.EMBEDDING_MODEL
    logger.info("Backfilling with default embedding model: %s", default_model)

    clients_updated, clients_skipped = backfill_clients(default_model)
    logger.info(
        "clients.embedding_model — updated: %d, already set: %d",
        clients_updated,
        clients_skipped,
    )

    registry_updated, registry_skipped = backfill_vector_registry(default_model)
    logger.info(
        "vector_node_registry.embedding_model — updated: %d, already set: %d",
        registry_updated,
        registry_skipped,
    )

    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()
