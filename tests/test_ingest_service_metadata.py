from dataclasses import dataclass
from datetime import datetime, timezone

from app.services.ingest_service import (
    _apply_metadata_exclusions,
    _apply_retrieval_metadata,
)


@dataclass
class DummyNode:
    node_id: str
    metadata: dict
    excluded_embed_metadata_keys: list[str] | None = None
    excluded_llm_metadata_keys: list[str] | None = None


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
    assert "document_id" in node.excluded_embed_metadata_keys
    assert "version_rank" in node.excluded_llm_metadata_keys
    assert node.excluded_embed_metadata_keys == sorted(
        node.excluded_embed_metadata_keys
    )
    assert node.excluded_llm_metadata_keys == sorted(node.excluded_llm_metadata_keys)
    assert len(node.excluded_embed_metadata_keys) == len(
        set(node.excluded_embed_metadata_keys)
    )
