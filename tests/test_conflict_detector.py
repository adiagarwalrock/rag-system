from llama_index.core.schema import NodeWithScore, TextNode

from app.retrieval.conflict_detector import detect_conflicts


def _node(node_id: str, text: str, metadata: dict, score: float = 0.8) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(id_=node_id, text=text, metadata=metadata), score=score
    )


def test_detects_numeric_disagreement_within_version_family():
    newer = _node(
        "newer",
        "Portfolio occupancy rate was 95% in the latest quarter.",
        {
            "document_name": "Operations v2",
            "document_id": "doc-v2",
            "version_label": "v2",
            "document_version_group": "ops_policy",
        },
    )
    older = _node(
        "older",
        "Portfolio occupancy rate was 91% in the previous quarter.",
        {
            "document_name": "Operations v1",
            "document_id": "doc-v1",
            "version_label": "v1",
            "document_version_group": "ops_policy",
        },
    )

    conflicts = detect_conflicts(
        [newer, older],
        evidence_nodes=[newer, older],
        question="What changed across versions?",
    )

    assert len(conflicts) == 1
    assert conflicts[0]["conflict_type"] == "numeric_disagreement"
    assert "95%" in conflicts[0]["summary"]
    assert "91%" in conflicts[0]["summary"]


def test_detects_cross_document_conflicts_when_question_is_cross_source():
    report_a = _node(
        "report-a",
        "Demand growth in suburban markets was 12% this year.",
        {
            "document_name": "Market Outlook A",
            "document_id": "doc-a",
            "version_label": "current",
            "document_version_group": "market_outlook_a",
        },
    )
    report_b = _node(
        "report-b",
        "Demand growth in suburban markets was 7% this year.",
        {
            "document_name": "Market Outlook B",
            "document_id": "doc-b",
            "version_label": "current",
            "document_version_group": "market_outlook_b",
        },
    )

    conflicts = detect_conflicts(
        [report_a, report_b],
        evidence_nodes=[report_a, report_b],
        question="Are there conflicting data points across documents?",
    )

    assert len(conflicts) == 1
    assert "12%" in conflicts[0]["summary"]
    assert "7%" in conflicts[0]["summary"]


def test_ignores_same_document_same_version_numeric_differences():
    node_a = _node(
        "same-doc-a",
        "Revenue reached $100 million in Q1.",
        {
            "document_name": "Single Source",
            "document_id": "doc-single",
            "version_label": "v2",
            "document_version_group": "single_group",
        },
    )
    node_b = _node(
        "same-doc-b",
        "Revenue reached $120 million in Q2.",
        {
            "document_name": "Single Source",
            "document_id": "doc-single",
            "version_label": "v2",
            "document_version_group": "single_group",
        },
    )

    conflicts = detect_conflicts(
        [node_a, node_b], evidence_nodes=[node_a, node_b], question="Find conflicts"
    )

    assert conflicts == []
