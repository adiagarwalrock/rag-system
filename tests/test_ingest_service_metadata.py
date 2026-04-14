from dataclasses import dataclass
from datetime import datetime, timezone

from llama_index.core.schema import NodeRelationship, TextNode

from app.services.ingest_service import (
    _apply_metadata_exclusions,
    _apply_ref_doc_ids,
    _apply_retrieval_metadata,
    _build_non_layout_node_parser,
)


@dataclass
class DummyNode:
    node_id: str
    metadata: dict
    excluded_embed_metadata_keys: list[str] | None = None
    excluded_llm_metadata_keys: list[str] | None = None
    ref_doc_id: str | None = None


def test_apply_retrieval_metadata_sets_labels_and_version_fields():
    node = DummyNode(
        node_id="node-1",
        metadata={"page_num": 3, "chunk_id": "chunk-abc"},
    )
    version_info = {
        "version_rank": 7,
        "published_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "is_current": True,
    }

    _apply_retrieval_metadata(
        [node], filename="handbook.pdf", version_info=version_info
    )

    assert node.metadata["chunk_id"] == "chunk-abc"
    assert node.metadata["source_file"] == "handbook.pdf"
    assert node.metadata["citation_label"] == "handbook.pdf - p.3 - chunk 1"
    assert node.metadata["version_rank"] == 7
    assert node.metadata["published_at"].startswith("2024-01-01")
    assert node.metadata["is_current"] is True


def test_apply_retrieval_metadata_generates_chunk_id_when_missing():
    node = DummyNode(node_id="node-2", metadata={"slide_num": 5})

    _apply_retrieval_metadata([node], filename="deck.pptx", version_info={})

    assert node.metadata["chunk_id"]
    assert node.metadata["citation_label"] == "deck.pptx - slide 5 - chunk 1"


def test_apply_metadata_exclusions_merges_existing_keys_without_duplicates():
    node = DummyNode(
        node_id="node-3",
        metadata={},
        excluded_embed_metadata_keys=["already_here", "document_id"],
        excluded_llm_metadata_keys=["already_here"],
    )

    _apply_metadata_exclusions([node])

    assert "already_here" in node.excluded_embed_metadata_keys
    assert "citation_label" in node.excluded_embed_metadata_keys
    assert "chunk_type" in node.excluded_embed_metadata_keys
    assert "source_artifact_type" in node.excluded_embed_metadata_keys
    assert "document_id" in node.excluded_embed_metadata_keys
    assert "version_rank" in node.excluded_llm_metadata_keys
    assert node.excluded_embed_metadata_keys == sorted(
        node.excluded_embed_metadata_keys
    )
    assert node.excluded_llm_metadata_keys == sorted(node.excluded_llm_metadata_keys)
    assert len(node.excluded_embed_metadata_keys) == len(
        set(node.excluded_embed_metadata_keys)
    )


def test_build_non_layout_node_parser_semantic_uses_config(monkeypatch):
    from app.core.config import settings
    from app.services import ingest_service

    calls: dict = {}

    def _fake_from_defaults(**kwargs):
        calls.update(kwargs)
        return "semantic-parser"

    monkeypatch.setattr(
        ingest_service.SemanticSplitterNodeParser,
        "from_defaults",
        staticmethod(_fake_from_defaults),
    )

    monkeypatch.setattr(settings, "SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE", 88)
    monkeypatch.setattr(settings, "SEMANTIC_SPLITTER_BUFFER_SIZE", 2)

    parser = _build_non_layout_node_parser()

    assert parser == "semantic-parser"
    assert calls["breakpoint_percentile_threshold"] == 88
    assert calls["buffer_size"] == 2
    assert "embed_model" in calls


def test_apply_ref_doc_ids_sets_from_metadata_document_id():
    first = TextNode(text="first", metadata={"document_id": "doc-1"})
    second = TextNode(text="second", metadata={"document_id": "None"})
    third = TextNode(text="third", metadata={})

    _apply_ref_doc_ids([first, second, third])

    assert first.ref_doc_id == "doc-1"
    assert first.relationships[NodeRelationship.SOURCE].node_id == "doc-1"
    assert second.ref_doc_id is None
    assert third.ref_doc_id is None
