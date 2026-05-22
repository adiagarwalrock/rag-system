from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf as fitz
from llama_index.core import Document as LlamaDocument

from app.core.config import settings
from app.ingestion.parser.custom.pdf_pipeline.adapters import DefaultPDFExtractionStage
from app.ingestion.parser.custom.pdf_pipeline.artifact_builders import DefaultArtifactStage
from app.ingestion.parser.custom.pdf_pipeline.chunk_builder import DefaultChunkStage
from app.ingestion.parser.custom.pdf_pipeline.contracts import (
    ArtifactStage,
    ChunkStage,
    PageStructureStage,
    PDFExtractionStage,
)
from app.ingestion.parser.custom.pdf_pipeline.page_structure import DefaultPageStructureStage
from app.ingestion.parser.custom.pdf_pipeline.registry import PDFPipelineRegistry
from app.ingestion.parser.custom.pdf_pipeline.repair import repair_pdf_path, suppress_mupdf_messages

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    artifact_root: Path
    screenshot_dir: Path
    figure_dir: Path
    raw_dir: Path

    @classmethod
    def for_document(cls, document_id: str) -> "PipelinePaths":
        artifact_root = Path(settings.PARSED_ARTIFACTS_DIR) / document_id
        return cls(
            artifact_root=artifact_root,
            screenshot_dir=artifact_root / "screenshots",
            figure_dir=artifact_root / "figures",
            raw_dir=artifact_root / "raw",
        )


def parse_pdf_layout_aware(
    file_path: str,
    document_metadata: dict[str, Any],
) -> tuple[list[LlamaDocument], list[dict[str, Any]]]:
    pipeline = PDFIngestionPipeline(
        file_path=file_path,
        document_metadata=document_metadata,
    )
    return pipeline.execute()


class PDFIngestionPipeline:
    def __init__(
        self,
        *,
        file_path: str,
        document_metadata: dict[str, Any],
        extraction_stage: PDFExtractionStage | None = None,
        page_structure_stage: PageStructureStage | None = None,
        artifact_stage: ArtifactStage | None = None,
        chunk_stage: ChunkStage | None = None,
    ) -> None:
        self.file_path = file_path
        self.document_metadata = document_metadata
        self.document_id = document_metadata.get("document_id") or str(uuid.uuid4())
        self.source_file = os.path.basename(file_path)
        self.paths = PipelinePaths.for_document(self.document_id)

        registry = PDFPipelineRegistry()
        self.extraction_stage = extraction_stage or registry.resolve_extraction_stage(
            DefaultPDFExtractionStage
        )
        self.page_structure_stage = (
            page_structure_stage
            or registry.resolve_page_structure_stage(DefaultPageStructureStage)
        )
        self.artifact_stage = artifact_stage or registry.resolve_artifact_stage(
            DefaultArtifactStage
        )
        self.chunk_stage = chunk_stage or registry.resolve_chunk_stage(
            DefaultChunkStage
        )

    def execute(self) -> tuple[list[LlamaDocument], list[dict[str, Any]]]:
        self._prepare_workspace()
        parse_input_path, repair_meta = self._prepare_parse_input()

        with fitz.open(parse_input_path) as pdf_doc:
            extraction = self.extraction_stage.extract(
                pdf_doc=pdf_doc,
                file_path=self.file_path,
                parse_input_path=parse_input_path,
                screenshot_dir=self.paths.screenshot_dir,
            )
            page_manifests, regions = self.page_structure_stage.build(
                document_id=self.document_id,
                liteparse_pages=extraction.liteparse_pages,
                pymupdf_pages=extraction.pymupdf_pages,
                parse_meta=extraction.parse_meta,
                repair_meta=repair_meta,
            )
            artifact_result = self.artifact_stage.build(
                pdf_doc=pdf_doc,
                page_manifests=page_manifests,
                pymupdf_pages=extraction.pymupdf_pages,
                regions=regions,
                figure_dir=self.paths.figure_dir,
            )

        chunk_artifacts = self.chunk_stage.build_chunks(
            page_manifests=page_manifests,
            regions=regions,
            tables=artifact_result.merged_tables,
            figures=artifact_result.figures,
            reasoning_artifacts=artifact_result.reasoning_artifacts,
        )

        self.chunk_stage.write_bundle(
            artifact_root=self.paths.artifact_root,
            liteparse_pages=extraction.liteparse_pages,
            pymupdf_pages=extraction.pymupdf_pages,
            page_manifests=page_manifests,
            regions=regions,
            table_fragments=artifact_result.table_fragments,
            merged_tables=artifact_result.merged_tables,
            figures=artifact_result.figures,
            chunks=chunk_artifacts,
            reasoning_artifacts=artifact_result.reasoning_artifacts,
        )

        docs, units = self.chunk_stage.to_llama_docs(
            chunk_artifacts=chunk_artifacts,
            document_metadata=self.document_metadata,
            source_file=self.source_file,
            artifact_root=self.paths.artifact_root,
        )
        logger.info(
            "Parsed %s via layout-aware PDF pipeline: %d chunks, %d units",
            self.source_file,
            len(docs),
            len(units),
        )
        return docs, units

    def _prepare_workspace(self) -> None:
        self.paths.artifact_root.mkdir(parents=True, exist_ok=True)
        self.paths.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figure_dir.mkdir(parents=True, exist_ok=True)
        self.paths.raw_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_parse_input(self) -> tuple[str, dict[str, Any]]:
        if getattr(settings, "SUPPRESS_MUPDF_STDERR", True):
            suppress_mupdf_messages()

        repair_meta = {
            "pdf_repair_attempted": False,
            "pdf_repair_method": None,
            "pdf_repair_success": False,
            "pdf_repair_error": None,
        }
        parse_input_path = self.file_path
        if getattr(settings, "ENABLE_PDF_REPAIR_PREPASS", True):
            parse_input_path, repair_meta = repair_pdf_path(self.file_path)
        return parse_input_path, repair_meta
