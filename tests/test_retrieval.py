from llama_index.core.schema import NodeWithScore, TextNode

from app.retrieval.citation_builder import build_citations
from app.retrieval.query_expansion import should_expand_query
from app.retrieval.retriever import VecteraRetriever, _fuse_node_batches


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
    evidence_doc_ids = {node.node.metadata.get("document_id") for node in evidence_nodes}

    assert "doc-a" in evidence_doc_ids
    assert "doc-b" in evidence_doc_ids
