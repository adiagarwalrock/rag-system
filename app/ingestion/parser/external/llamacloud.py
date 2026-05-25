"""
LlamaParse external parser.

Provides LlamaParseParser — a class with parse / extract / run methods:
  parse()   → ParsedDocument  (page chunks with text + base metadata from MarkdownPageAnalyzer)
  extract() → DocumentExtraction  (same chunks with provider-enriched ChunkMetadata)
  run()     → tuple[list[LlamaDocument], list[dict]]  (same shape as parse_pdf_layout_aware())

The module-level run() shim preserves backward-compatible call sites.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import time
from typing import Any

from app.core.config import settings
from app.ingestion.parser.external.helper import (
    DocumentExtraction,
    MarkdownPageAnalyzer,
    ParsedDocument,
    ParsedPageChunk,
    ProviderInfo,
    build_metadata_association_prompt,
    normalize_metadata_associations,
    provider_metadata_association_schema,
    provider_usage,
    split_by_page_markers,
    to_llama_docs_from_extraction,
)
from llama_index.core import Document as LlamaDocument

from app.ingestion.parser.external.prompts import (
    LLAMA_CLOUD_AGENTIC_AUTO_MODE_PARSING_PROMPT,
    LLAMA_CLOUD_AGENT_CUSTOM_PROMPT,
    LLAMA_CLOUD_EXTRACTION_PROMPT,
)

class LlamaParseParser:
    """
    Wraps the LlamaParse API: parse → extract → run.
    Output is identical in structure to parse_pdf_layout_aware().

    Class attributes
    ----------------
    PARSER_NAME       : str — "llamaparse"
    LAYOUT_ENGINE     : str — "llamaparse_vlm"
    PARSER_VERSION    : str — from settings.EXTERNAL_PARSER_VERSION
    EXTRACT_TIER      : str — "agentic"
    EXTRACT_TIMEOUT_S : int — 120
    """

    PARSER_NAME: str = "llamaparse"
    LAYOUT_ENGINE: str = "llamaparse_vlm"
    PARSER_VERSION: str = settings.EXTERNAL_PARSER_VERSION
    EXTRACT_TIER: str = "agentic"
    EXTRACT_TIMEOUT_S: int = 120

    def __init__(self) -> None:
        from llama_cloud import LlamaCloud

        self._client = LlamaCloud(api_key=settings.LLAMAPARSE_API_KEY)
        self._analyzer = MarkdownPageAnalyzer(
            parser_name=self.PARSER_NAME,
            layout_engine=self.LAYOUT_ENGINE,
        )

    # ------------------------------------------------------------------
    # parse
    # ------------------------------------------------------------------

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Upload + parse via LlamaParse API; return a ParsedDocument with base metadata."""
        start = time.perf_counter()

        # Step 1: upload
        print(f"  [llamaparse] uploading {pdf_path.name} ...")
        file = self._client.files.create(file=str(pdf_path), purpose="parse")

        # Step 2: parse (SDK blocks until the job finishes)
        print("  [llamaparse] parsing (agentic_plus, blocking) ...")
        result = self._client.parsing.parse(
            file_id=file.id,
            tier="agentic_plus",
            version="latest",
            expand=["markdown"],
            output_options={
                "markdown": {
                    "tables": {
                        "output_tables_as_markdown": True,
                        "merge_continued_tables": True,
                        "compact_markdown_tables": True,
                    },
                    "annotate_links": True,
                    "inline_images": True,
                },
                "spatial_text": {
                    "do_not_unroll_columns": True,
                    "preserve_very_small_text": True,
                },
            },
            processing_options={
                "specialized_chart_parsing": "agentic_plus",
                "aggressive_table_extraction": True,
                "ocr_parameters": {
                    "languages": ["en"],
                },
                "auto_mode_configuration": [
                    {
                        "page_contains_at_least_n_charts": 1,
                        "parsing_conf": {
                            "tier": "agentic_plus",
                            "version": "latest",
                            "specialized_chart_parsing": "agentic_plus",
                            "custom_prompt": LLAMA_CLOUD_AGENTIC_AUTO_MODE_PARSING_PROMPT,
                        },
                    },
                ],
            },
            agentic_options={
                "custom_prompt": LLAMA_CLOUD_AGENT_CUSTOM_PROMPT,
            },
        )

        elapsed = time.perf_counter() - start

        pages = result.markdown.pages if result.markdown else []
        print(f"  [llamaparse] done — {len(pages)} pages extracted ({elapsed:.1f}s)")

        # Build markdown with page markers
        parts: list[str] = []
        for p in pages:
            if not (hasattr(p, "markdown") and p.markdown):
                continue
            page_num = p.page_number if hasattr(p, "page_number") else None
            if page_num is not None:
                parts.append(f"[[START OF PAGE {page_num}]]")
            parts.append(p.markdown)
            if page_num is not None:
                parts.append(f"[[END OF PAGE {page_num}]]")
        markdown = "\n\n".join(parts)

        # Extract parse job ID for LlamaCloud Extract.
        # ParsingGetResponse nests the job under result.job.id (not result.id).
        job_obj = getattr(result, "job", None)
        parse_job_id = (
            getattr(job_obj, "id", None)
            or getattr(result, "id", None)
            or getattr(result, "job_id", None)
        )

        # Build ParsedPageChunks from page-sectioned markdown
        page_sections = split_by_page_markers(markdown)
        page_chunks = self._build_page_chunks(page_sections)

        return ParsedDocument(
            source_file=str(pdf_path),
            parser_name=self.PARSER_NAME,
            parser_version=self.PARSER_VERSION,
            parse_job_id=str(parse_job_id) if parse_job_id else None,
            extract_input_id=None,
            page_chunks=page_chunks,
        )

    def _build_page_chunks(
        self, page_sections: list[tuple[int, str]]
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
                    asset_refs=analyzed.asset_refs,
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------

    def extract(self, parsed_document: ParsedDocument) -> DocumentExtraction:
        """Call LlamaCloud Extract API to enrich chunk metadata. Returns DocumentExtraction."""
        if not parsed_document.parse_job_id:
            raise ValueError(
                "LlamaCloud Extract requires a LlamaParse parse_job_id. "
                "Ensure parse() succeeded and stored the job ID."
            )

        start = time.perf_counter()
        print(
            f"  [llama extract] extracting metadata from "
            f"{parsed_document.parse_job_id} ({len(parsed_document.page_chunks)} chunks) ..."
        )

        job = self._client.extract.run(
            file_input=parsed_document.parse_job_id,
            configuration={
                "data_schema": provider_metadata_association_schema(),
                "tier": self.EXTRACT_TIER,
                "extract_version": "latest",
                "extraction_target": "per_page",
                "cite_sources": True,
                "confidence_scores": True,
                "system_prompt": build_metadata_association_prompt(
                    LLAMA_CLOUD_EXTRACTION_PROMPT, parsed_document
                ),
            },
            verbose=True,
        )

        elapsed = time.perf_counter() - start
        extraction = normalize_metadata_associations(
            job.extract_result,
            ProviderInfo(
                provider="llama",
                elapsed_s=round(elapsed, 2),
                job_id=getattr(job, "id", None),
                usage=provider_usage(self._get_nested(job, "metadata", "usage")),
                raw_citations=self._llama_citations(job),
            ),
            parsed_document,
        )
        print(f"  [llama extract] done — {len(extraction.chunks)} chunks enriched")
        return extraction

    @staticmethod
    def _llama_citations(job: Any) -> list[Any]:
        metadata = getattr(job, "extract_metadata", None)
        if metadata is None:
            return []
        from app.ingestion.parser.external.helper import to_plain_data

        plain = to_plain_data(metadata)
        if isinstance(plain, dict):
            citations = (
                plain.get("citations")
                or plain.get("sources")
                or plain.get("field_metadata")
            )
            return (
                citations
                if isinstance(citations, list)
                else ([citations] if citations else [])
            )
        return []

    @staticmethod
    def _get_nested(value: Any, first: str, second: str) -> Any:
        parent = getattr(value, first, None)
        if isinstance(parent, dict):
            return parent.get(second)
        return getattr(parent, second, None)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(
        self,
        pdf_path: Path,
        document_metadata: dict[str, Any],
    ) -> tuple[list[LlamaDocument], list[dict[str, Any]]]:
        """Parse + extract + convert to (docs, units). Same shape as parse_pdf_layout_aware()."""
        parsed = self.parse(pdf_path)
        extracted = self.extract(parsed)
        return to_llama_docs_from_extraction(extracted, document_metadata)


# ---------------------------------------------------------------------------
# Module-level shim — preserves existing call sites in parse_document()
# ---------------------------------------------------------------------------


def run(
    pdf_path: Path,
    document_metadata: dict[str, Any],
) -> tuple[list[LlamaDocument], list[dict[str, Any]]]:
    """Parse a PDF via LlamaParse and return (docs, units) with full metadata parity."""
    return LlamaParseParser().run(pdf_path, document_metadata)
