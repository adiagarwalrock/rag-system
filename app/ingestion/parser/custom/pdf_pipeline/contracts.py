from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf as fitz
from llama_index.core import Document as LlamaDocument

from app.ingestion.parser.custom.pdf_pipeline.models import (
    ChunkArtifact,
    FigureArtifact,
    PageManifest,
    ReasoningArtifact,
    Region,
    TableArtifact,
)


@dataclass(slots=True)
class ExtractionResult:
    liteparse_pages: list[dict[str, Any]]
    pymupdf_pages: list[dict[str, Any]]
    parse_meta: dict[str, Any]


@dataclass(slots=True)
class ArtifactResult:
    table_fragments: list[TableArtifact]
    merged_tables: list[TableArtifact]
    figures: list[FigureArtifact]
    reasoning_artifacts: list[ReasoningArtifact] = field(default_factory=list)


class PDFExtractionStage(ABC):
    @abstractmethod
    def extract(
        self,
        *,
        pdf_doc: fitz.Document,
        file_path: str,
        parse_input_path: str,
        screenshot_dir: Path,
    ) -> ExtractionResult:
        raise NotImplementedError


class PageStructureStage(ABC):
    @abstractmethod
    def build(
        self,
        *,
        document_id: str,
        liteparse_pages: list[dict[str, Any]],
        pymupdf_pages: list[dict[str, Any]],
        parse_meta: dict[str, Any],
        repair_meta: dict[str, Any],
    ) -> tuple[list[PageManifest], list[Region]]:
        raise NotImplementedError


class ArtifactStage(ABC):
    @abstractmethod
    def build(
        self,
        *,
        pdf_doc: fitz.Document,
        page_manifests: list[PageManifest],
        pymupdf_pages: list[dict[str, Any]],
        regions: list[Region],
        figure_dir: Path,
    ) -> ArtifactResult:
        raise NotImplementedError


class ChunkStage(ABC):
    @abstractmethod
    def build_chunks(
        self,
        *,
        page_manifests: list[PageManifest],
        regions: list[Region],
        tables: list[TableArtifact],
        figures: list[FigureArtifact],
        reasoning_artifacts: list[ReasoningArtifact] | None = None,
    ) -> list[ChunkArtifact]:
        raise NotImplementedError

    @abstractmethod
    def write_bundle(
        self,
        *,
        artifact_root: Path,
        liteparse_pages: list[dict[str, Any]],
        pymupdf_pages: list[dict[str, Any]],
        page_manifests: list[PageManifest],
        regions: list[Region],
        table_fragments: list[TableArtifact],
        merged_tables: list[TableArtifact],
        figures: list[FigureArtifact],
        chunks: list[ChunkArtifact],
        reasoning_artifacts: list[ReasoningArtifact] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def to_llama_docs(
        self,
        *,
        chunk_artifacts: list[ChunkArtifact],
        document_metadata: dict[str, Any],
        source_file: str,
        artifact_root: Path,
    ) -> tuple[list[LlamaDocument], list[dict[str, Any]]]:
        raise NotImplementedError
