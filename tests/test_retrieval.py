from llama_index.core.schema import NodeWithScore, TextNode

from app.retrieval.citation_builder import build_citations
from app.retrieval.query_expansion import should_expand_query
from app.retrieval.retriever import (
    VecteraRetriever,
    _build_retrieval_diagnostics,
    _collect_image_evidence_paths,
    _fuse_node_batches,
    _split_reasoning_from_text,
)


def _node(node_id: str, score: float, metadata: dict | None = None) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            id_=node_id, text=f"source text for {node_id}", metadata=metadata or {}
        ),
        score=score,
    )


def test_query_expansion_routes_broad_and_version_questions_only():
    assert should_expand_query("Compare the current version against the older version")
    assert should_expand_query("Summarize trends across the uploaded reports")
    assert should_expand_query("Show me the chart for quarterly revenue")
    assert should_expand_query("How did customer count change over time?")
    assert not should_expand_query("What is the renewal deadline?")


def test_fuse_node_batches_dedupes_and_boosts_repeated_nodes():
    first = _node("same", 0.4)
    second = _node("other", 0.7)
    duplicate = _node("same", 0.5)

    fused = _fuse_node_batches([[first, second], [duplicate]])

    assert [node.node.node_id for node in fused].count("same") == 1
    assert fused[0].node.node_id == "same"
    assert fused[0].score > second.score


def test_latest_query_prefers_current_version_nodes():
    retriever = VecteraRetriever("client-1", top_k=5)

    current = _node(
        "current",
        0.5,
        {
            "version_rank": "12",
            "is_current": "true",
            "version_label": "v2",
            "document_version_group": "policy",
            "document_id": "doc-v2",
        },
    )
    old = _node(
        "old",
        0.5,
        {
            "version_rank": "1",
            "is_current": "false",
            "version_label": "v1",
            "document_version_group": "policy",
            "document_id": "doc-v1",
        },
    )

    ranked = retriever._rank_nodes("What is the latest policy version?", [old, current])
    assert ranked[0].node.node_id == "current"


def test_citations_are_bounded_subset_of_ranked_candidates():
    retriever = VecteraRetriever("client-1", top_k=10)
    ranked = [
        _node(
            f"node-{i}",
            1.0 - (i * 0.01),
            {
                "version_label": "v2",
                "document_id": f"doc-{i}",
                "citation_label": f"Doc {i}",
            },
        )
        for i in range(10)
    ]

    evidence_nodes = retriever._select_evidence_nodes(
        "What is the renewal deadline?", ranked
    )
    citations = build_citations(evidence_nodes)

    assert len(ranked) == 10
    assert len(citations) <= retriever.evidence_limit

    ranked_ids = {node.node.node_id for node in ranked}
    citation_ids = {citation["vector_node_id"] for citation in citations}
    assert citation_ids <= ranked_ids


def test_comparison_queries_select_multiple_versions_for_citations():
    retriever = VecteraRetriever("client-1", top_k=10)
    nodes = [
        _node(
            "v2-primary",
            0.92,
            {
                "version_rank": "12",
                "is_current": "true",
                "version_label": "v2",
                "document_version_group": "policy",
                "document_id": "doc-v2",
            },
        ),
        _node(
            "v2-secondary",
            0.88,
            {
                "version_rank": "12",
                "is_current": "true",
                "version_label": "v2",
                "document_version_group": "policy",
                "document_id": "doc-v2-2",
            },
        ),
        _node(
            "v1-primary",
            0.84,
            {
                "version_rank": "10",
                "is_current": "false",
                "version_label": "v1",
                "document_version_group": "policy",
                "document_id": "doc-v1",
            },
        ),
    ]

    ranked = retriever._rank_nodes("Compare changes between v1 and v2", nodes)
    evidence_nodes = retriever._select_evidence_nodes(
        "Compare changes between v1 and v2", ranked
    )
    citations = build_citations(evidence_nodes)
    versions = {citation.get("version_label") for citation in citations}

    assert "v1" in versions
    assert "v2" in versions


def test_comparison_query_deprioritizes_reasoning_chunks_for_factual_delta_questions():
    retriever = VecteraRetriever("client-1", top_k=10)
    ranked = [
        _node(
            "reasoning-v2",
            0.95,
            {
                "chunk_type": "reasoning_chart",
                "version_label": "v2",
                "document_id": "doc-v2",
                "document_version_group": "policy",
            },
        ),
        _node(
            "reasoning-v1",
            0.93,
            {
                "chunk_type": "reasoning_chart",
                "version_label": "v1",
                "document_id": "doc-v1",
                "document_version_group": "policy",
            },
        ),
        _node(
            "factual-v2",
            0.9,
            {
                "chunk_type": "body_text",
                "version_label": "v2",
                "document_id": "doc-v2",
                "document_version_group": "policy",
            },
        ),
        _node(
            "factual-v1",
            0.88,
            {
                "chunk_type": "body_text",
                "version_label": "v1",
                "document_id": "doc-v1",
                "document_version_group": "policy",
            },
        ),
    ]

    evidence_nodes = retriever._select_evidence_nodes(
        "How much did customer count change between v1 and v2?",
        ranked,
    )

    selected_ids = [node.node.node_id for node in evidence_nodes]
    assert selected_ids[:2] == ["factual-v2", "factual-v1"]


def test_retrieval_diagnostics_reports_reasoning_and_document_diversity():
    ranked = [
        _node(
            "reasoning-1",
            0.8,
            {"chunk_type": "reasoning_chart", "document_id": "doc-1"},
        ),
        _node("body-1", 0.7, {"chunk_type": "body_text", "document_id": "doc-1"}),
        _node("body-2", 0.6, {"chunk_type": "body_text", "document_id": "doc-2"}),
    ]
    evidence = [ranked[1], ranked[2]]

    diagnostics = _build_retrieval_diagnostics(ranked, evidence)

    assert diagnostics["ranked_document_count"] == 2
    assert diagnostics["evidence_document_count"] == 2
    assert diagnostics["ranked_reasoning_count"] == 1
    assert diagnostics["evidence_reasoning_count"] == 0
    assert diagnostics["ranked_image_chunk_count"] == 0
    assert diagnostics["evidence_image_chunk_count"] == 0


def test_conflict_query_diversifies_evidence_when_versions_are_missing():
    retriever = VecteraRetriever("client-1", top_k=10)
    ranked = [
        _node(
            "doc-a-primary",
            0.95,
            {
                "document_id": "doc-a",
                "document_version_group": "group-a",
                "contains_numeric_data": True,
            },
        ),
        _node(
            "doc-a-secondary",
            0.92,
            {
                "document_id": "doc-a",
                "document_version_group": "group-a",
                "contains_numeric_data": True,
            },
        ),
        _node(
            "doc-b-primary",
            0.9,
            {
                "document_id": "doc-b",
                "document_version_group": "group-b",
                "contains_numeric_data": True,
            },
        ),
    ]

    evidence_nodes = retriever._select_evidence_nodes(
        "Are there conflicting data points across documents?", ranked
    )
    evidence_doc_ids = {
        node.node.metadata.get("document_id") for node in evidence_nodes
    }

    assert "doc-a" in evidence_doc_ids
    assert "doc-b" in evidence_doc_ids


def test_build_citations_includes_asset_refs_and_visual_metadata():
    citations = build_citations(
        [
            _node(
                "figure-1",
                0.8,
                {
                    "chunk_type": "figure_artifact",
                    "asset_refs": ["/tmp/fake-chart.png"],
                    "figure_type": "chart",
                    "chart_type": "line",
                    "table_id": None,
                },
            )
        ]
    )

    assert len(citations) == 1
    assert citations[0]["asset_refs"] == ["/tmp/fake-chart.png"]
    assert citations[0]["has_image_assets"] is True
    assert citations[0]["figure_type"] == "chart"
    assert citations[0]["chart_type"] == "line"


def test_build_citations_dedupes_asset_refs_across_citations():
    citations = build_citations(
        [
            _node(
                "figure-1",
                0.8,
                {
                    "chunk_type": "figure_artifact",
                    "asset_refs": ["/tmp/shared-image.png"],
                },
            ),
            _node(
                "figure-2",
                0.7,
                {
                    "chunk_type": "chart_context",
                    "asset_refs": ["/tmp/shared-image.png"],
                },
            ),
        ]
    )

    assert citations[0]["asset_refs"] == ["/tmp/shared-image.png"]
    assert citations[1]["asset_refs"] == []


def test_collect_image_evidence_paths_resolves_relative_and_dedupes(tmp_path):
    absolute_image = tmp_path / "absolute.png"
    absolute_image.write_bytes(b"fake")

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    relative_image = bundle_dir / "relative.png"
    relative_image.write_bytes(b"fake")
    non_image = bundle_dir / "note.txt"
    non_image.write_text("not an image", encoding="utf-8")

    citations = [
        {
            "asset_refs": [str(absolute_image), str(absolute_image)],
            "artifact_bundle_path": None,
        },
        {
            "asset_refs": ["relative.png", "note.txt"],
            "artifact_bundle_path": str(bundle_dir),
        },
    ]

    paths = _collect_image_evidence_paths(citations, max_images=10)

    # Same image bytes should be de-duplicated even when paths differ.
    assert paths == [str(absolute_image.resolve())]


def test_collect_image_evidence_paths_includes_unique_image_content(tmp_path):
    first_image = tmp_path / "first.png"
    first_image.write_bytes(b"first")
    second_image = tmp_path / "second.png"
    second_image.write_bytes(b"second")

    citations = [
        {"asset_refs": [str(first_image)], "artifact_bundle_path": None},
        {"asset_refs": [str(second_image)], "artifact_bundle_path": None},
    ]

    paths = _collect_image_evidence_paths(citations, max_images=10)

    assert paths == [str(first_image.resolve()), str(second_image.resolve())]


def test_visual_query_injects_image_evidence_when_top_evidence_has_no_images():
    retriever = VecteraRetriever("client-1", top_k=10)
    ranked = [
        _node(
            f"text-{i}",
            1.0 - (i * 0.01),
            {
                "chunk_type": "body_text",
                "document_id": f"doc-{i}",
            },
        )
        for i in range(7)
    ]
    ranked.extend(
        [
            _node(
                "image-1",
                0.5,
                {
                    "chunk_type": "figure_artifact",
                    "document_id": "img-doc-1",
                    "asset_refs": ["/tmp/figure-1.png"],
                },
            ),
            _node(
                "image-2",
                0.49,
                {
                    "chunk_type": "chart_context",
                    "document_id": "img-doc-2",
                    "asset_refs": ["/tmp/figure-2.png"],
                },
            ),
        ]
    )

    evidence = retriever._select_evidence_nodes("What does the chart image show?", ranked)
    image_evidence = [node for node in evidence if node.node.metadata.get("asset_refs")]

    assert len(evidence) == retriever.evidence_limit
    assert len(image_evidence) >= 2


def test_split_reasoning_from_text_extracts_thinking_and_answer():
    raw = (
        "<thinking>Check Source [1] and [2], compare figures, reconcile conflicts.</thinking>\n"
        "<answer>Revenue rises from 10 to 14 across versions [1][2].</answer>"
    )

    answer, reasoning = _split_reasoning_from_text(raw)

    assert answer == "Revenue rises from 10 to 14 across versions [1][2]."
    assert reasoning is not None
    assert "compare figures" in reasoning
