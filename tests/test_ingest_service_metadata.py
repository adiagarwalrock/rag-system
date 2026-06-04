from dataclasses import dataclass
from datetime import datetime, timezone

from llama_index.core import Document as LlamaDocument
from llama_index.core.schema import NodeRelationship, TextNode

from app.services.ingest_service import (
    _apply_metadata_exclusions,
    _apply_ref_doc_ids,
    _apply_retrieval_metadata,
    _build_non_layout_node_parser,
    IngestionPipelineExecutor,
)
from app.ingestion.parser.external.helper.chunk_schema import DocumentExtraction
from app.ingestion.parser.external.helper.chunk_schema import to_llama_docs_from_extraction
from app.ingestion.retrieval_metadata import (
    CANONICAL_RETRIEVAL_METADATA_FIELDS,
    normalize_retrieval_metadata,
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
    assert node.metadata["version_label"] is None
    assert node.metadata["document_version_group"] is None


def test_apply_retrieval_metadata_generates_chunk_id_when_missing():
    node = DummyNode(node_id="node-2", metadata={"slide_num": 5})

    _apply_retrieval_metadata([node], filename="deck.pptx", version_info={})

    assert node.metadata["chunk_id"]
    assert node.metadata["citation_label"] == "deck.pptx - slide 5 - chunk 1"


def test_normalize_retrieval_metadata_fills_canonical_contract():
    metadata = normalize_retrieval_metadata(
        {
            "page_num": "3",
            "page_nums": ["3"],
            "version_rank": "202603",
            "is_current": "false",
            "asset_refs": "chart.png",
            "parser_name": "rag_parser",
        },
        text="5,500+ Customers 232,500 Cross Connects 55+ Metros",
        document_metadata={
            "document_id": "doc-1",
            "client_id": "client-1",
            "document_name": "Digital Realty March 2026.pdf",
            "file_name": "Digital Realty March 2026.pdf",
        },
        source_file="Digital Realty March 2026.pdf",
        chunk_index=2,
    )

    for field in CANONICAL_RETRIEVAL_METADATA_FIELDS:
        assert field in metadata
    assert metadata["page_num"] == 3
    assert metadata["page_nums"] == [3]
    assert metadata["version_rank"] == 202603
    assert metadata["is_current"] is False
    assert metadata["contains_numeric_data"] is True
    assert metadata["asset_refs"] == ["chart.png"]
    assert metadata["citation_label"] == "Digital Realty March 2026.pdf - p.3 - chunk 2"


def test_external_parser_conversion_emits_canonical_metadata_contract():
    extraction = DocumentExtraction.model_validate(
        {
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "chunk_type": "body_text",
                    "page_nums": [17],
                    "text": "5,500+ Customers 232,500 Cross Connects 55+ Metros",
                    "metadata": {
                        "parser_name": "reducto",
                        "parser_version": "1.0.0",
                        "document_date": "2026-03",
                    },
                }
            ]
        }
    )

    docs, units = to_llama_docs_from_extraction(
        extraction,
        {
            "document_id": "doc-1",
            "client_id": "client-1",
            "document_name": "deck.pdf",
            "file_name": "deck.pdf",
        },
    )

    metadata = docs[0].metadata
    for field in CANONICAL_RETRIEVAL_METADATA_FIELDS:
        assert field in metadata
    assert metadata["parser_name"] == "reducto"
    assert metadata["page_num"] == 17
    assert metadata["contains_numeric_data"] is True
    assert units[0]["document_date"] == "2026-03"


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


def test_text_chunk_type_still_uses_semantic_splitter(monkeypatch):
    from app.services import ingest_service

    executor = IngestionPipelineExecutor.__new__(IngestionPipelineExecutor)
    executor.filename = "legacy.txt"

    captured: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, transformations):
            captured["transformations"] = transformations

        def run(self, documents, num_workers):
            captured["documents"] = documents
            captured["num_workers"] = num_workers
            return [TextNode(text="split", metadata={"chunk_type": "body_text"})]

    monkeypatch.setattr(
        ingest_service, "_build_non_layout_node_parser", lambda **_: "splitter"
    )
    monkeypatch.setattr(ingest_service, "IngestionPipeline", FakePipeline)
    monkeypatch.setattr(ingest_service, "TitleExtractor", lambda nodes: "title")
    monkeypatch.setattr(
        ingest_service, "SummaryExtractor", lambda summaries: "summary"
    )
    monkeypatch.setattr(ingest_service, "KeywordExtractor", lambda keywords: "keyword")
    monkeypatch.setattr(
        ingest_service,
        "QuestionsAnsweredExtractor",
        lambda num_questions: "questions",
    )

    nodes = executor._run_ingestion_pipeline(
        llama_docs=[LlamaDocument(text="legacy text", metadata={"chunk_type": "text"})]
    )

    assert captured["documents"][0].metadata["chunk_type"] == "text"
    assert captured["num_workers"] == 4
    assert nodes[0].text == "split"


def test_apply_ref_doc_ids_sets_from_metadata_document_id():
    first = TextNode(text="first", metadata={"document_id": "doc-1"})
    second = TextNode(text="second", metadata={"document_id": "None"})
    third = TextNode(text="third", metadata={})

    _apply_ref_doc_ids([first, second, third])

    assert first.ref_doc_id == "doc-1"
    assert first.relationships[NodeRelationship.SOURCE].node_id == "doc-1"
    assert second.ref_doc_id is None
    assert third.ref_doc_id is None
