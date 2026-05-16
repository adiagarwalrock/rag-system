"""Table artifact operations split module."""

from app.ingestion.pdf_pipeline.artifact.core import (
    build_table_artifacts,
    merge_table_artifacts,
)

__all__ = ["build_table_artifacts", "merge_table_artifacts"]
