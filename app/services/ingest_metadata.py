import uuid
from typing import Any

from llama_index.core import Settings as LlamaSettings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo

from app.core.config import settings
from app.indexing.vector_store import vector_store_manager

NON_SEMANTIC_EMBED_METADATA_KEYS = (
    "document_id",
    "doc_id",
    "ref_doc_id",
    "client_id",
    "client_name",
    "ingestion_job_id",
    "parser_name",
    "parser_version",
    "file_name",
    "source_file",
    "file_type",
    "chunk_id",
    "chunk_type",
    "citation_label",
    "version_label",
    "document_version_group",
    "version_rank",
    "published_at",
    "effective_from",
    "effective_to",
    "is_current",
    "authority_score",
    "table_detected",
    "chart_detected",
    "contains_numeric_data",
    "page_num",
    "page_nums",
    "slide_num",
    "section_path",
    "region_ids",
    "source_artifact_type",
    "source_artifact_id",
    "bbox_refs",
    "caption",
    "numeric_density",
    "layout_confidence",
    "complexity_score",
    "asset_refs",
    "parser_sources",
    "page_class",
    "artifact_bundle_path",
    "table_title",
    "figure_type",
    "chart_type",
    "chart_title",
    "x_axis_label",
    "y_axis_label",
    "x_categories",
    "series",
    "approx_datapoints",
    "trend_summary",
    "key_chart_facts",
    "numeric_extraction_confidence",
    "chart_parse_status",
    "llm_caption_model",
    "llm_caption_version",
    "llm_caption_prompt_version",
    "llm_caption_status",
    "llm_caption_error",
    "units",
    "continuation_flag",
    "ocr_used",
    "table_id",
    "reasoning_type",
    "source_artifact_ids",
    "evidence_refs",
    "reasoning_confidence",
    "reasoning_model",
    "reasoning_prompt_version",
    "claims",
    "llm_enriched",
)

NON_SEMANTIC_LLM_METADATA_KEYS = (
    "document_id",
    "doc_id",
    "ref_doc_id",
    "client_id",
    "client_name",
    "ingestion_job_id",
    "parser_name",
    "parser_version",
    "file_type",
    "chunk_id",
    "chunk_type",
    "citation_label",
    "version_rank",
    "published_at",
    "is_current",
    "page_num",
    "page_nums",
    "source_artifact_type",
    "source_artifact_id",
    "chart_type",
    "chart_title",
    "chart_parse_status",
    "llm_caption_status",
    "asset_refs",
    "artifact_bundle_path",
    "reasoning_type",
    "source_artifact_ids",
    "evidence_refs",
    "reasoning_confidence",
    "reasoning_model",
    "reasoning_prompt_version",
    "claims",
    "llm_enriched",
)

_MISSING_DOC_ID_SENTINELS = {"", "none", "null", "n/a", "na", "undefined"}


def _build_non_layout_node_parser() -> Any:
    embed_model = getattr(LlamaSettings, "_embed_model", None)
    if embed_model is None and not settings.is_openai_api_key_placeholder:
        vector_store_manager.configure_llama_settings()
        embed_model = getattr(LlamaSettings, "_embed_model", None)

    return SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        breakpoint_percentile_threshold=settings.SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE,
        buffer_size=settings.SEMANTIC_SPLITTER_BUFFER_SIZE,
    )


def _apply_metadata_exclusions(nodes: list[BaseNode]) -> None:
    for node in nodes:
        embed_excluded = set(node.excluded_embed_metadata_keys or [])
        llm_excluded = set(node.excluded_llm_metadata_keys or [])

        embed_excluded.update(NON_SEMANTIC_EMBED_METADATA_KEYS)
        llm_excluded.update(NON_SEMANTIC_LLM_METADATA_KEYS)

        node.excluded_embed_metadata_keys = sorted(embed_excluded)
        node.excluded_llm_metadata_keys = sorted(llm_excluded)


def _normalize_document_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in _MISSING_DOC_ID_SENTINELS:
            return None
        return normalized
    return str(value)


def _apply_ref_doc_ids(nodes: list[BaseNode]) -> None:
    """Ensure top-level vector-store doc IDs are derived from canonical metadata."""
    for node in nodes:
        metadata = node.metadata or {}
        document_id = _normalize_document_id(metadata.get("document_id"))
        if document_id:
            relationships = dict(node.relationships or {})
            relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=document_id
            )
            node.relationships = relationships


def _isoformat_or_none(value: Any) -> str | None:
    return value.isoformat() if value else None


def _core_version_metadata(version_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "version_rank": version_info.get("version_rank", 0),
        "published_at": _isoformat_or_none(version_info.get("published_at")),
        "is_current": version_info.get("is_current", False),
    }


def _document_version_metadata(version_info: dict[str, Any]) -> dict[str, Any]:
    return {
        **_core_version_metadata(version_info),
        "version_label": version_info.get("version_label"),
        "document_version_group": version_info.get("version_group"),
        "effective_from": _isoformat_or_none(version_info.get("effective_from")),
        "effective_to": _isoformat_or_none(version_info.get("effective_to")),
    }


def _apply_retrieval_metadata(
    nodes: list[BaseNode],
    filename: str,
    version_info: dict[str, Any],
) -> None:
    for index, node in enumerate(nodes, start=1):
        metadata = node.metadata or {}
        chunk_id = metadata.get("chunk_id") or str(uuid.uuid4())
        page_num = metadata.get("page_num")
        slide_num = metadata.get("slide_num")

        label_parts = [filename]
        if page_num:
            label_parts.append(f"p.{page_num}")
        if slide_num:
            label_parts.append(f"slide {slide_num}")
        label_parts.append(f"chunk {index}")

        metadata.update(
            {
                "chunk_id": chunk_id,
                "source_file": metadata.get("source_file") or filename,
                "citation_label": metadata.get("citation_label")
                or " - ".join(label_parts),
                **_core_version_metadata(version_info),
            }
        )
        node.metadata = metadata
