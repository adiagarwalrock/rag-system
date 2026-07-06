"""
Reducto external parser.

Provides ReductoParser — a class with parse / extract / run methods:
  parse()   → ParsedDocument  (page chunks with text + base metadata from MarkdownPageAnalyzer)
  extract() → DocumentExtraction  (same chunks with provider-enriched ChunkMetadata)
  run()     → tuple[list[LlamaDocument], list[dict]]  (same shape as parse_pdf_layout_aware())

The module-level run() shim preserves backward-compatible call sites.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.ingestion.parser.external.helper import (
    DocumentExtraction,
    MarkdownPageAnalyzer,
    ParsedDocument,
    ParsedPageChunk,
    ProviderInfo,
    build_metadata_association_prompt,
    normalize_metadata_associations,
    provider_citations,
    provider_metadata_association_schema,
    provider_usage,
    render_page_screenshots,
    screenshot_refs,
    split_by_page_markers,
    to_llama_docs_from_extraction,
)
from llama_index.core import Document as LlamaDocument

from app.ingestion.parser.external.prompts import (
    REDUCTO_TABLE_PARSER_PROMPT,
    REDUCTO_CHART_PARSER_PROMPT,
    REDUCTO_EXTRACTION_PROMPT,
)

class ReductoParser:
    """
    Wraps the Reducto API: parse → extract → run.
    Output is identical in structure to parse_pdf_layout_aware().

    Class attributes
    ----------------
    PARSER_NAME    : str — "reducto"
    LAYOUT_ENGINE  : str — "reducto_vlm"
    PARSER_VERSION : str — from settings.EXTERNAL_PARSER_VERSION
    """

    PARSER_NAME: str = "reducto"
    LAYOUT_ENGINE: str = "reducto_vlm"
    PARSER_VERSION: str = settings.EXTERNAL_PARSER_VERSION

    def __init__(self) -> None:
        from reducto import Reducto

        self._client = Reducto(api_key=settings.REDUCTO_API_KEY)
        self._analyzer = MarkdownPageAnalyzer(
            parser_name=self.PARSER_NAME,
            layout_engine=self.LAYOUT_ENGINE,
        )

    # ------------------------------------------------------------------
    # parse
    # ------------------------------------------------------------------

    def parse(self, pdf_path: Path, document_id: str | None = None) -> ParsedDocument:
        """Upload + parse via Reducto API; return a ParsedDocument with base metadata."""
        start = time.perf_counter()

        # Step 1: upload
        print(f"  [reducto] uploading {pdf_path.name} ...")
        with open(pdf_path, "rb") as f:
            upload = self._client.upload(file=(pdf_path.name, f, "application/pdf"))

        # Step 2: parse
        print("  [reducto] parsing (agentic_plus chart + table + text agents) ...")
        response = self._client.parse.run(
            input=upload.file_id,
            async_={"priority": True},
            formatting={
                "table_output_format": "md",
                "merge_tables": True,
                "add_page_markers": True,
                "include": ["hyperlinks"],
            },
            enhance={
                "summarize_figures": True,
                "intelligent_ordering": True,
                "agentic": [
                    {
                        "scope": "table",
                        "prompt": REDUCTO_TABLE_PARSER_PROMPT,
                    },
                    {
                        "scope": "figure",
                        "advanced_chart_agent": True,
                        "prompt": REDUCTO_CHART_PARSER_PROMPT,
                    },
                ],
            },
            retrieval={
                "chunking": {
                    "chunk_mode": "page_sections",
                },
            },
        )

        elapsed = time.perf_counter() - start
        print(
            f"  [reducto] done — result type: {response.result.type} ({elapsed:.1f}s)"
        )

        # Collect chunks from Reducto
        if response.result.type == "url":
            chunks = httpx.get(response.result.url, timeout=60).json().get("chunks", [])
        else:
            chunks = response.result.chunks

        # Build markdown from chunks
        parts: list[str] = []
        current_page: int | None = None
        for c in chunks:
            content = c.content if hasattr(c, "content") else c.get("content", "")
            if not content:
                continue
            try:
                blocks = c.blocks if hasattr(c, "blocks") else c.get("blocks", [])
                if blocks:
                    first_block = blocks[0]
                    bbox = (
                        first_block.bbox
                        if hasattr(first_block, "bbox")
                        else first_block.get("bbox", {})
                    )
                    page = bbox.page if hasattr(bbox, "page") else bbox.get("page")
                else:
                    page = None
            except (AttributeError, IndexError, KeyError):
                page = None

            if page is not None and page != current_page:
                if current_page is not None:
                    parts.append(f"[[END OF PAGE {current_page}]]")
                parts.append(f"[[START OF PAGE {page}]]")
                current_page = page
            parts.append(content)

        if current_page is not None:
            parts.append(f"[[END OF PAGE {current_page}]]")

        markdown = "\n\n".join(parts)

        # Post-process: strip collapsed base-parser rows (>500 char pipe rows)
        lines = markdown.splitlines()
        clean_lines: list[str] = []
        skip_next_separator = False
        for line in lines:
            if line.startswith("|") and len(line) > 500:
                skip_next_separator = True
                continue
            if skip_next_separator and re.match(r"^\|[-|]+\|?$", line):
                skip_next_separator = False
                continue
            skip_next_separator = False
            clean_lines.append(line)
        markdown = "\n".join(clean_lines)
        markdown = markdown.replace("\\n", "\n")

        # Extract job_id for Reducto Extract
        # Reducto Extract uses a jobid:// reference into the parse result
        job_id = getattr(response, "job_id", None) or getattr(upload, "file_id", None)
        extract_input_id = f"jobid://{job_id}" if job_id else None

        # Build ParsedPageChunks from page-sectioned markdown
        page_sections = split_by_page_markers(markdown)
        screenshot_map = render_page_screenshots(pdf_path, document_id, page_sections)
        page_chunks = self._build_page_chunks(page_sections, screenshot_map)

        return ParsedDocument(
            source_file=str(pdf_path),
            parser_name=self.PARSER_NAME,
            parser_version=self.PARSER_VERSION,
            parse_job_id=None,
            extract_input_id=extract_input_id,
            page_chunks=page_chunks,
        )

    def _build_page_chunks(
        self,
        page_sections: list[tuple[int, str]],
        screenshot_map: dict[int, str] | None = None,
    ) -> list[ParsedPageChunk]:
        chunks: list[ParsedPageChunk] = []
        for page_num, content in page_sections:
            if not content.strip():
                continue
            analyzed = self._analyzer.analyze(page_num, content)
            analyzed.metadata["parser_version"] = self.PARSER_VERSION
            chunks.append(
                ParsedPageChunk(
                    chunk_id=analyzed.chunk_id,
                    page_nums=analyzed.page_nums,
                    text=analyzed.text,
                    chunk_type=analyzed.chunk_type,
                    source_artifact_type=analyzed.source_artifact_type,
                    source_artifact_id=analyzed.source_artifact_id,
                    metadata=analyzed.metadata,
                    asset_refs=screenshot_refs(analyzed.page_nums, screenshot_map),
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------

    def extract(self, parsed_document: ParsedDocument) -> DocumentExtraction:
        """Call Reducto Extract API to enrich chunk metadata. Returns DocumentExtraction."""
        if not parsed_document.extract_input_id:
            raise ValueError(
                "Reducto Extract requires a jobid:// parse extract input id. "
                "Ensure parse() succeeded and stored the job ID."
            )

        start = time.perf_counter()
        print(
            f"  [reducto extract] extracting metadata from "
            f"{parsed_document.extract_input_id} ({len(parsed_document.page_chunks)} chunks) ..."
        )

        response = self._client.extract.run(
            input=parsed_document.extract_input_id,
            instructions={
                "schema": provider_metadata_association_schema(),
                "system_prompt": build_metadata_association_prompt(
                    REDUCTO_EXTRACTION_PROMPT, parsed_document
                ),
            },
            settings={
                "array_extract": True,
                "deep_extract": True,
                "include_images": False,  # skip per-page VLM image pass — saves ~150-250s
                "citations": {
                    "enabled": True,
                    "numerical_confidence": False,  # skip per-value confidence pass — saves ~50s
                },
            },
        )

        elapsed = time.perf_counter() - start
        extraction = normalize_metadata_associations(
            response.result,
            ProviderInfo(
                provider="reducto",
                elapsed_s=round(elapsed, 2),
                job_id=getattr(response, "job_id", None),
                studio_link=getattr(response, "studio_link", None),
                usage=provider_usage(getattr(response, "usage", None)),
                raw_citations=provider_citations(getattr(response, "citations", None)),
            ),
            parsed_document,
        )
        print(f"  [reducto extract] done — {len(extraction.chunks)} chunks enriched")
        return extraction

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(
        self,
        pdf_path: Path,
        document_metadata: dict[str, Any],
    ) -> tuple[list[LlamaDocument], list[dict[str, Any]]]:
        """Parse + extract + convert to (docs, units). Same shape as parse_pdf_layout_aware()."""
        document_id = document_metadata.get("document_id")
        parsed = self.parse(pdf_path, document_id=document_id)
        extracted = self.extract(parsed)
        return to_llama_docs_from_extraction(extracted, document_metadata)


# ---------------------------------------------------------------------------
# Module-level shim — preserves existing call sites in parse_document()
# ---------------------------------------------------------------------------


def run(
    pdf_path: Path,
    document_metadata: dict[str, Any],
) -> tuple[list[LlamaDocument], list[dict[str, Any]]]:
    """Parse a PDF via Reducto and return (docs, units) with full metadata parity."""
    return ReductoParser().run(pdf_path, document_metadata)
