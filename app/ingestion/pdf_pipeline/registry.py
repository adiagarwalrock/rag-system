from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, TypeVar

from app.ingestion.pdf_pipeline.contracts import (
    ArtifactStage,
    ChunkStage,
    PDFExtractionStage,
    PageStructureStage,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SingletonMeta(type):
    _instances: dict[type[Any], Any] = {}
    _instances_lock = threading.RLock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        with cls._instances_lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class LayoutEngineManager(metaclass=SingletonMeta):
    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._state_lock = threading.Lock()
        self._layout_activated = False
        self._layout_failed = False

    def ensure_layout_activation(self) -> bool:
        with self._state_lock:
            if self._layout_activated:
                return True
            if self._layout_failed:
                return False

            try:
                import pymupdf.layout  # noqa: F401
            except Exception:
                self._layout_failed = True
                logger.warning(
                    "pymupdf.layout is unavailable; continuing with native extraction"
                )
                return False

            self._layout_activated = True
            return True

    def reset(self) -> None:
        with self._state_lock:
            self._layout_activated = False
            self._layout_failed = False


class ChunkerManager(metaclass=SingletonMeta):
    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._state_lock = threading.Lock()
        self._loaded = False
        self._sentence_chunker_cls: Any | None = None
        self._table_chunker_cls: Any | None = None

    def create_sentence_chunker(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Any | None:
        self._load_chunkers()
        if self._sentence_chunker_cls is None:
            return None
        return self._sentence_chunker_cls(
            tokenizer="character",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def create_table_chunker(self, *, chunk_size: int) -> Any | None:
        self._load_chunkers()
        if self._table_chunker_cls is None:
            return None
        return self._table_chunker_cls(tokenizer="row", chunk_size=chunk_size)

    def reset(self) -> None:
        with self._state_lock:
            self._loaded = False
            self._sentence_chunker_cls = None
            self._table_chunker_cls = None

    def _load_chunkers(self) -> None:
        with self._state_lock:
            if self._loaded:
                return

            self._loaded = True
            try:
                from chonkie import SentenceChunker, TableChunker
            except Exception:  # pragma: no cover
                return

            self._sentence_chunker_cls = SentenceChunker
            self._table_chunker_cls = TableChunker


class PDFPipelineRegistry(metaclass=SingletonMeta):
    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._lock = threading.Lock()
        self._extraction_stage: PDFExtractionStage | None = None
        self._page_structure_stage: PageStructureStage | None = None
        self._artifact_stage: ArtifactStage | None = None
        self._chunk_stage: ChunkStage | None = None
        self.layout_engine_manager = LayoutEngineManager()
        self.chunker_manager = ChunkerManager()

    def set_extraction_stage(self, stage: PDFExtractionStage) -> None:
        with self._lock:
            self._extraction_stage = stage

    def set_page_structure_stage(self, stage: PageStructureStage) -> None:
        with self._lock:
            self._page_structure_stage = stage

    def set_artifact_stage(self, stage: ArtifactStage) -> None:
        with self._lock:
            self._artifact_stage = stage

    def set_chunk_stage(self, stage: ChunkStage) -> None:
        with self._lock:
            self._chunk_stage = stage

    def resolve_extraction_stage(
        self, factory: Callable[[], PDFExtractionStage]
    ) -> PDFExtractionStage:
        return self._resolve("_extraction_stage", factory)

    def resolve_page_structure_stage(
        self, factory: Callable[[], PageStructureStage]
    ) -> PageStructureStage:
        return self._resolve("_page_structure_stage", factory)

    def resolve_artifact_stage(
        self, factory: Callable[[], ArtifactStage]
    ) -> ArtifactStage:
        return self._resolve("_artifact_stage", factory)

    def resolve_chunk_stage(self, factory: Callable[[], ChunkStage]) -> ChunkStage:
        return self._resolve("_chunk_stage", factory)

    def reset(self) -> None:
        with self._lock:
            self._extraction_stage = None
            self._page_structure_stage = None
            self._artifact_stage = None
            self._chunk_stage = None
        self.layout_engine_manager.reset()
        self.chunker_manager.reset()

    def _resolve(self, attr_name: str, factory: Callable[[], T]) -> T:
        with self._lock:
            current = getattr(self, attr_name)
            if current is None:
                current = factory()
                setattr(self, attr_name, current)
            return current
