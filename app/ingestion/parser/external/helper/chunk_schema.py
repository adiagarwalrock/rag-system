"""
chunk_schema.py — Shared Pydantic models and pipeline functions for external parsers.

This module is the canonical home for:
  - ParsedDocument / ParsedPageChunk  (intermediate parse output)
  - ChunkMetadata / ExtractedChunk / DocumentExtraction  (final chunk models)
  - ProviderMetadataAssociationExtraction  (schema sent to provider Extract APIs)
  - normalize_metadata_associations()  (merges provider metadata onto parsed chunks)
  - build_metadata_association_prompt()  (helper for building Extract API prompts)
  - to_llama_docs_from_extraction()  (convert DocumentExtraction → (docs, units))

Provider-specific Extract API calls (prompts, API parameters, response handling)
live in each parser's own file: reducto.py and llamacloud.py.

The extractor's sole responsibility is metadata enrichment — it never changes
chunk text, chunk_type, or page_nums produced by parse().
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from llama_index.core import Document as LlamaDocument
from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.parser.custom.pdf_pipeline.helpers import (
    has_numeric_data,
    numeric_density,
)


# ===========================================================================
# ParsedDocument / ParsedPageChunk — intermediate parse output
# ===========================================================================


@dataclass
class ParsedPageChunk:
    """One page section produced by parse(); text and chunk_type are immutable."""

    chunk_id: str
    page_nums: list[int]
    text: str
    chunk_type: str                  # set by MarkdownPageAnalyzer, never overwritten
    source_artifact_type: str
    source_artifact_id: str
    metadata: dict[str, Any]
    asset_refs: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Full parse output for one document, ready for the Extract API."""

    source_file: str
    parser_name: str
    parser_version: str
    parse_job_id: str | None        # LlamaParse pjb-... ID used by LlamaExtract
    extract_input_id: str | None    # Reducto jobid:// reference used by Reducto Extract
    page_chunks: list[ParsedPageChunk]


# ===========================================================================
# Extraction Pydantic models (mirrored from sample_extractor.py)
# ===========================================================================


class Citation(BaseModel):
    """Provider-grounded evidence for a chunk-level extraction."""

    model_config = ConfigDict(extra="allow")

    text: str = Field(default="", description="Quoted or paraphrased source evidence.")
    page_num: int | None = Field(
        default=None, description="One-based page number for the cited evidence."
    )
    source: str = Field(default="", description="Provider source, block, or citation id.")
    confidence: float | None = Field(
        default=None, description="Provider confidence for this citation, if supplied."
    )


class ChunkMetadata(BaseModel):
    """Metadata fields expected by the downstream chunk model.

    Fields produced by MarkdownPageAnalyzer (parse step) are populated first.
    Fields marked 'Extract API' are left None by the analyzer and enriched by
    the provider Extract API call in extract().
    """

    model_config = ConfigDict(extra="allow")

    # -- layout / position ------------------------------------------------
    section_path: str = Field(default="")
    region_ids: list[str] = Field(default_factory=list)
    bbox_refs: list[str] = Field(default_factory=list)
    caption: str | None = Field(default=None)
    numeric_density: float | None = Field(default=None)
    layout_confidence: float | None = Field(default=None)
    complexity_score: float | None = Field(default=None)
    asset_refs: list[str] = Field(default_factory=list)
    parser_sources: list[str] = Field(default_factory=list)
    page_class: str | None = Field(default=None)
    layout_engine: str | None = Field(default=None)
    page_parse_degraded: bool | None = Field(default=None)
    degraded_stages: list[str] = Field(default_factory=list)
    pdf_repair_attempted: bool | None = Field(default=None)
    pdf_repair_method: str | None = Field(default=None)
    pdf_repair_success: bool | None = Field(default=None)
    pdf_repair_error: str | None = Field(default=None)
    llm_page_summary_status: str | None = Field(default=None)
    llm_page_summary_error: str | None = Field(default=None)
    ocr_used: bool | None = Field(default=None)
    units: list[str] = Field(default_factory=list)
    continuation_flag: bool | None = Field(default=None)
    figure_type: str | None = Field(default=None)
    source_artifact_type: str | None = Field(default=None)

    # -- table metadata ---------------------------------------------------
    table_id: str | None = Field(default=None)
    table_title: str | None = Field(default=None)
    llm_enriched: bool | None = Field(default=None)
    llm_enrichment_confidence: float | None = Field(default=None)

    # -- chart metadata (Extract API) -------------------------------------
    chart_type: str | None = Field(default=None)
    chart_title: str | None = Field(default=None)
    x_axis_label: str | None = Field(default=None)
    y_axis_label: str | None = Field(default=None)
    x_categories: list[str] = Field(default_factory=list)
    series: list[dict[str, Any]] = Field(default_factory=list)
    approx_datapoints: list[dict[str, Any]] = Field(default_factory=list)
    trend_summary: str | None = Field(default=None)
    key_chart_facts: list[str] = Field(default_factory=list)
    numeric_extraction_confidence: float | None = Field(default=None)
    chart_parse_status: str | None = Field(default=None)
    llm_caption_model: str | None = Field(default=None)
    llm_caption_version: str | None = Field(default=None)
    llm_caption_prompt_version: str | None = Field(default=None)
    llm_caption_status: str | None = Field(default=None)
    llm_caption_error: str | None = Field(default=None)

    # -- reasoning metadata -----------------------------------------------
    reasoning_type: str | None = Field(default=None)
    source_artifact_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    reasoning_confidence: float | None = Field(default=None)
    reasoning_model: str | None = Field(default=None)
    reasoning_prompt_version: str | None = Field(default=None)
    claims: list[dict[str, Any]] = Field(default_factory=list)

    # -- document / chunk identity ----------------------------------------
    source_file: str | None = Field(default=None)
    page_num: int | None = Field(default=None)
    page_nums: list[int] = Field(default_factory=list)
    chunk_id: str | None = Field(default=None)
    chunk_type: str | None = Field(default=None)
    source_artifact_id: str | None = Field(default=None)
    artifact_bundle_path: str | None = Field(default=None)
    parser_name: str | None = Field(default=None)
    parser_version: str | None = Field(default=None)
    table_detected: bool | None = Field(default=None)
    chart_detected: bool | None = Field(default=None)
    contains_numeric_data: bool | None = Field(default=None)

    # -- ingest metadata --------------------------------------------------
    document_id: str | None = Field(default=None)
    client_id: str | None = Field(default=None)
    client_name: str | None = Field(default=None)
    document_name: str | None = Field(default=None)
    file_name: str | None = Field(default=None)
    file_type: str | None = Field(default=None)
    ingestion_job_id: str | None = Field(default=None)

    # -- temporal / scope (Extract API) -----------------------------------
    document_date: str | None = Field(
        default=None,
        description="ISO document date (YYYY-MM-DD, YYYY-MM, YYYY, or YYYY-Q#). Extract API only.",
    )
    as_of_date: str | None = Field(
        default=None,
        description="ISO as-of date for metric snapshot. Extract API only.",
    )
    metric_basis: str | None = Field(
        default=None,
        description="Metric basis: 'actual', 'guidance', 'pro_forma', 'estimate', 'target'. Extract API only.",
    )


class ExtractedChunk(BaseModel):
    """One fully-enriched chunk after parse + extract."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str = Field(default="")
    chunk_type: Literal[
        "body_text", "full_table", "table_segment", "table_summary_text",
        "figure_artifact", "visual_proxy_text", "chart_context", "chart_data_points",
        "page_card", "reasoning", "text", "table", "figure", "chart", "other",
    ] = Field(default="body_text")
    source_artifact_type: str | None = Field(default=None)
    source_artifact_id: str | None = Field(default=None)
    page_nums: list[int] = Field(default_factory=list)
    text: str = Field(default="")
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    asset_refs: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    extraction_confidence: float | None = Field(default=None)


class ProviderInfo(BaseModel):
    """Provider execution details retained outside the extraction schema."""

    model_config = ConfigDict(extra="allow")

    provider: str = Field(default="")
    elapsed_s: float = Field(default=0.0)
    job_id: str | None = Field(default=None)
    studio_link: str | None = Field(default=None)
    usage: dict[str, Any] = Field(default_factory=dict)
    raw_citations: list[Any] = Field(default_factory=list)
    parse_job_id: str | None = Field(default=None)
    extract_input_id: str | None = Field(default=None)
    parser_name: str | None = Field(default=None)
    parser_version: str | None = Field(default=None)


class ProviderDocumentMetadata(BaseModel):
    """Provider-facing document metadata."""

    model_config = ConfigDict(extra="forbid")

    document_id: str | None = Field(default=None)
    client_id: str | None = Field(default=None)
    client_name: str | None = Field(default=None)
    document_name: str | None = Field(default=None)
    file_name: str | None = Field(default=None)
    file_type: str | None = Field(default=None)
    ingestion_job_id: str | None = Field(default=None)
    source_file: str | None = Field(default=None)
    parser_name: str | None = Field(default=None)
    parser_version: str | None = Field(default=None)


class ChunkMetadataAssociation(BaseModel):
    """Provider-facing metadata association for an existing parser page chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(
        default="",
        description="Existing parser chunk id. Return the same id from the prompt.",
    )
    page_nums: list[int] = Field(default_factory=list)
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    asset_refs: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    extraction_confidence: float | None = Field(default=None)


class ProviderMetadataAssociationExtraction(BaseModel):
    """Provider-facing schema for metadata-only extraction over parser chunks."""

    model_config = ConfigDict(extra="forbid")

    document_metadata: ProviderDocumentMetadata = Field(
        default_factory=ProviderDocumentMetadata
    )
    chunks: list[ChunkMetadataAssociation] = Field(default_factory=list)


class DocumentExtraction(BaseModel):
    """Structured extraction output for one document."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    document_metadata: dict[str, Any] = Field(default_factory=dict)
    chunks: list[ExtractedChunk] = Field(default_factory=list)
    provider: ProviderInfo = Field(
        default_factory=ProviderInfo,
        alias="_provider",
    )


# ===========================================================================
# Schema helpers
# ===========================================================================


def provider_metadata_association_schema() -> dict[str, Any]:
    """Return the JSON schema for the ProviderMetadataAssociationExtraction model."""
    return _remove_untyped_array_items(
        ProviderMetadataAssociationExtraction.model_json_schema()
    )


def _remove_untyped_array_items(schema: Any) -> Any:
    if isinstance(schema, dict):
        cleaned = {k: _remove_untyped_array_items(v) for k, v in schema.items()}
        if cleaned.get("type") == "array" and cleaned.get("items") == {}:
            cleaned["items"] = {"type": "object"}
        if cleaned.get("additionalProperties") is True:
            cleaned.pop("additionalProperties")
        return cleaned
    if isinstance(schema, list):
        return [_remove_untyped_array_items(item) for item in schema]
    return schema


# ===========================================================================
# Plain-data conversion helpers
# ===========================================================================


def to_plain_data(value: Any) -> Any:
    """Convert SDK model objects into JSON-serializable Python containers."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, Mapping):
        return {str(k): to_plain_data(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_plain_data(v) for v in value]
    return value


def provider_usage(value: Any) -> dict[str, Any]:
    plain = to_plain_data(value)
    return plain if isinstance(plain, dict) else {}


def provider_citations(value: Any) -> list[Any]:
    plain = to_plain_data(value)
    return plain if isinstance(plain, list) else []


# ===========================================================================
# normalize_metadata_associations — merges provider output onto parsed chunks
# ===========================================================================


def build_metadata_association_prompt(base_prompt: str, parsed_document: ParsedDocument) -> str:
    """Append the chunk list to a provider-specific base prompt.

    Each parser class defines its own base_prompt and calls this helper to
    produce the full system prompt for its Extract API call.
    """
    chunk_lines = [
        f"- chunk_id: {chunk.chunk_id}; page_nums: {chunk.page_nums}"
        for chunk in parsed_document.page_chunks
    ]
    return (
        base_prompt.strip()
        + "\n\nExisting parser chunks:\n"
        + "\n".join(chunk_lines)
    )


def normalize_metadata_associations(
    payload: Any,
    provider_info: ProviderInfo,
    parsed_document: ParsedDocument,
) -> DocumentExtraction:
    """Merge provider metadata associations onto parser-produced page chunks.

    Text, chunk_type, and page_nums from parsed_document are always preserved.
    Only metadata fields from the provider response are merged in — empty /
    None provider fields do NOT overwrite non-None analyzer values.
    """
    extracted = _coerce_document(to_plain_data(payload))

    associations_by_chunk_id: dict[str, list[ExtractedChunk]] = {}
    associations_by_page: dict[int, list[ExtractedChunk]] = {}
    for chunk in extracted.chunks:
        if chunk.chunk_id:
            associations_by_chunk_id.setdefault(chunk.chunk_id, []).append(chunk)
        for page_num in chunk.page_nums:
            associations_by_page.setdefault(page_num, []).append(chunk)

    merged_chunks: list[dict[str, Any]] = []
    for parsed_chunk in parsed_document.page_chunks:
        associations = associations_by_chunk_id.get(parsed_chunk.chunk_id)
        if not associations and parsed_chunk.page_nums:
            associations = associations_by_page.get(parsed_chunk.page_nums[0], [])
        merged_chunks.append(_merge_page_chunk(parsed_chunk, associations or []))

    provider_info.parse_job_id = parsed_document.parse_job_id
    provider_info.parser_name = parsed_document.parser_name
    provider_info.parser_version = parsed_document.parser_version
    provider_info.extract_input_id = parsed_document.extract_input_id

    return DocumentExtraction.model_validate(
        {
            "document_metadata": {
                **(extracted.document_metadata or {}),
                "source_file": parsed_document.source_file,
                "parser_name": parsed_document.parser_name,
                "parser_version": parsed_document.parser_version,
            },
            "chunks": merged_chunks,
            "_provider": provider_info.model_dump(mode="json"),
        }
    )


def _coerce_document(data: Any) -> DocumentExtraction:
    if isinstance(data, dict):
        if "chunks" in data:
            return DocumentExtraction.model_validate(data)
        if "document_metadata" in data and "chunks" not in data:
            return DocumentExtraction.model_validate({**data, "chunks": []})
        if _looks_like_chunk(data):
            return DocumentExtraction.model_validate({"chunks": [data]})
        if "result" in data:
            return _coerce_document(data["result"])
    if isinstance(data, list):
        if not data:
            return DocumentExtraction()
        if all(isinstance(item, dict) and "chunks" in item for item in data):
            document_metadata: dict[str, Any] = {}
            chunks: list[dict[str, Any]] = []
            for item in data:
                document_metadata.update(item.get("document_metadata") or {})
                chunks.extend(item.get("chunks") or [])
            return DocumentExtraction.model_validate(
                {"document_metadata": document_metadata, "chunks": chunks}
            )
        if all(isinstance(item, dict) and _looks_like_chunk(item) for item in data):
            return DocumentExtraction.model_validate({"chunks": data})
        if len(data) == 1:
            return _coerce_document(data[0])
    return DocumentExtraction()


def _looks_like_chunk(data: dict[str, Any]) -> bool:
    return any(key in data for key in ("chunk_id", "chunk_type", "text", "page_nums"))


def _merge_page_chunk(
    parsed_chunk: ParsedPageChunk,
    associations: list[ExtractedChunk],
) -> dict[str, Any]:
    """Merge provider metadata onto a parsed chunk. Text and chunk_type are immutable."""
    metadata = dict(parsed_chunk.metadata)
    asset_refs = list(parsed_chunk.asset_refs)
    citations: list[Any] = []
    confidences: list[float] = []

    for association in associations:
        # Only merge non-empty provider metadata; preserve analyzer values
        provider_meta = _clean_metadata(
            association.metadata.model_dump(mode="json", exclude_unset=True)
        )
        # Fields set by the analyzer must not be overwritten by the provider
        _ANALYZER_OWNED = {
            "chunk_type", "chunk_id", "source_artifact_type", "source_artifact_id",
            "page_num", "page_nums", "parser_name", "parser_version",
            "layout_engine", "parser_sources",
        }
        for key, value in provider_meta.items():
            if key not in _ANALYZER_OWNED:
                metadata[key] = value

        asset_refs.extend(getattr(association, "asset_refs", []) or [])
        citations.extend(getattr(association, "citations", []) or [])
        confidence = getattr(association, "extraction_confidence", None)
        if isinstance(confidence, int | float):
            confidences.append(float(confidence))

    confidence_value = (
        round(sum(confidences) / len(confidences), 4) if confidences else None
    )

    # Ensure chunk_type is never corrupted — always use the parsed value
    return {
        "chunk_id": parsed_chunk.chunk_id,
        "chunk_type": parsed_chunk.chunk_type,  # immutable: from MarkdownPageAnalyzer
        "source_artifact_type": parsed_chunk.source_artifact_type,
        "source_artifact_id": parsed_chunk.source_artifact_id,
        "page_nums": parsed_chunk.page_nums,
        "text": parsed_chunk.text,  # immutable: from parse()
        "metadata": metadata,
        "asset_refs": _dedupe_strings(asset_refs),
        "citations": [to_plain_data(c) for c in citations],
        "extraction_confidence": confidence_value,
    }


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and value != "" and value != [] and value != {}
    }


def _dedupe_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


# ===========================================================================
# to_llama_docs_from_extraction — final conversion to (docs, units)
# ===========================================================================

_TABLE_CHUNK_TYPES = {"full_table", "table_segment", "table_summary_text"}
_CHART_CHUNK_TYPES = {"figure_artifact", "chart_context", "visual_proxy_text", "chart_data_points"}


def to_llama_docs_from_extraction(
    extraction: DocumentExtraction,
    document_metadata: dict[str, Any],
) -> tuple[list[LlamaDocument], list[dict[str, Any]]]:
    """Convert a DocumentExtraction into (docs, units) matching artifact_chunks_to_llama_docs().

    This is the same shape as the custom pipeline's output:
      - docs: list[LlamaDocument] with full metadata payloads
      - units: list[dict] with fields used by ingest_service._apply_document_metadata()
    """
    docs: list[LlamaDocument] = []
    units: list[dict[str, Any]] = []

    for chunk in extraction.chunks:
        text = chunk.text.strip()
        if not text:
            continue

        # Start from the ChunkMetadata model, then layer in identifiers + signals
        meta = chunk.metadata.model_dump(mode="json", exclude_none=False)

        # Document-level metadata overwrites any defaults from ChunkMetadata
        meta.update(document_metadata)

        # Computed signals from chunk type and text content
        table_detected = chunk.chunk_type in _TABLE_CHUNK_TYPES
        chart_detected = chunk.chunk_type in _CHART_CHUNK_TYPES
        contains_numeric = has_numeric_data(text)

        nd = float(meta.get("numeric_density") or 0.0)
        nd = max(nd, numeric_density(text))

        meta.update({
            # chunk identity
            "chunk_id":              chunk.chunk_id,
            "chunk_type":            chunk.chunk_type,
            "source_artifact_type":  chunk.source_artifact_type,
            "source_artifact_id":    chunk.source_artifact_id,
            "page_nums":             chunk.page_nums,
            "page_num":              chunk.page_nums[0] if chunk.page_nums else 1,
            "asset_refs":            chunk.asset_refs,
            "artifact_bundle_path":  meta.get("artifact_bundle_path") or "",
            # computed signals
            "table_detected":        table_detected,
            "chart_detected":        chart_detected,
            "contains_numeric_data": contains_numeric,
            "numeric_density":       nd,
        })

        docs.append(LlamaDocument(text=text, metadata=meta))

        units.append({
            "id":                     chunk.chunk_id,
            "document_id":            document_metadata.get("document_id"),
            "unit_type":              chunk.chunk_type,
            "page_num":               meta["page_num"],
            "page_nums":              chunk.page_nums,
            "raw_text":               text,
            "table_detected":         table_detected,
            "chart_detected":         chart_detected,
            "contains_numeric_data":  contains_numeric,
            "chunk_type":             chunk.chunk_type,
            "source_artifact_type":   chunk.source_artifact_type,
            "source_artifact_id":     chunk.source_artifact_id,
            "section_path":           meta.get("section_path"),
            "layout_confidence":      meta.get("layout_confidence"),
            "complexity_score":       meta.get("complexity_score"),
            "artifact_bundle_path":   meta.get("artifact_bundle_path", ""),
            "layout_engine":          meta.get("layout_engine"),
            "pdf_repair_attempted":   False,
            "pdf_repair_method":      None,
            "pdf_repair_success":     False,
            "llm_enriched":           meta.get("llm_enriched", True),
            "degraded_stages":        [],
            "page_parse_degraded":    False,
            # temporal / scope fields populated by Extract API
            "document_date":          meta.get("document_date"),
            "as_of_date":             meta.get("as_of_date"),
            "metric_basis":           meta.get("metric_basis"),
        })

    return docs, units
