import json

from ui.components.qdrant_inspector import (
    _group_points_by_document,
    _normalize_point_item,
    _parse_pasted_points,
)


def _sample_payload(top_level_document_id: str = "None") -> dict:
    nested_metadata = {
        "document_id": "46882477-afe3-4ec9-9790-250987a857ed",
        "document_name": "Digital Realty_Investor Presentation December 2025.pdf",
        "file_name": "Digital Realty_Investor Presentation December 2025.pdf",
        "source_file": "Digital Realty_Investor Presentation December 2025.pdf",
        "client_id": "b24ca25a-4723-43c8-8051-e9b2f6aa650c",
        "chunk_id": "be692da8-a412-447c-9548-7ded509ddd9d",
        "chunk_type": "body_text",
        "page_num": 25,
        "page_nums": [25],
        "section_path": "Social Content Platform",
        "citation_label": "Digital Realty_Investor Presentation December 2025.pdf - p.25 - chunk 160",
    }
    node_content = {
        "id_": "000e089b-2d83-4643-90cc-703dbbf34150",
        "metadata": nested_metadata,
        "text": "Social Content Platform",
    }
    return {
        "document_id": top_level_document_id,
        "document_name": "Digital Realty_Investor Presentation December 2025.pdf",
        "file_name": "Digital Realty_Investor Presentation December 2025.pdf",
        "source_file": "Digital Realty_Investor Presentation December 2025.pdf",
        "page_num": 25,
        "page_nums": [25],
        "chunk_id": "be692da8-a412-447c-9548-7ded509ddd9d",
        "chunk_type": "body_text",
        "_node_type": "Document",
        "_node_content": json.dumps(node_content),
        "client_id": "b24ca25a-4723-43c8-8051-e9b2f6aa650c",
    }


def test_parse_pasted_points_accepts_single_object_and_points_array():
    single = _parse_pasted_points(json.dumps({"document_id": "doc-1"}))
    assert len(single) == 1
    assert single[0]["document_id"] == "doc-1"

    wrapped = _parse_pasted_points(
        json.dumps({"points": [{"document_id": "doc-2"}, {"document_id": "doc-3"}]})
    )
    assert [item["document_id"] for item in wrapped] == ["doc-2", "doc-3"]


def test_normalize_prefers_nested_document_id_over_top_level_none():
    point = {"payload": _sample_payload(top_level_document_id="None")}
    normalized = _normalize_point_item(point, index=1)

    assert normalized is not None
    assert normalized["document_id"] == "46882477-afe3-4ec9-9790-250987a857ed"
    assert normalized["document_key"] == "46882477-afe3-4ec9-9790-250987a857ed"
    assert normalized["page_num"] == 25
    assert normalized["page_nums"] == [25]


def test_group_points_merges_points_with_same_canonical_document():
    p1 = _normalize_point_item({"payload": _sample_payload("None")}, index=1)
    payload_2 = _sample_payload("None")
    payload_2["chunk_id"] = "chunk-2"
    payload_2["page_num"] = 26
    payload_2["page_nums"] = [26]
    node = json.loads(payload_2["_node_content"])
    node["metadata"]["chunk_id"] = "chunk-2"
    node["metadata"]["page_num"] = 26
    node["metadata"]["page_nums"] = [26]
    payload_2["_node_content"] = json.dumps(node)
    p2 = _normalize_point_item({"payload": payload_2}, index=2)

    grouped = _group_points_by_document([p1, p2])  # type: ignore[list-item]
    assert len(grouped) == 1
    assert grouped[0]["point_count"] == 2
    assert grouped[0]["page_count"] == 2


def test_group_points_falls_back_to_name_then_unknown():
    name_only = _normalize_point_item(
        {
            "payload": {
                "document_id": "None",
                "document_name": "Fallback Name",
                "chunk_id": "x1",
            }
        },
        index=1,
    )
    unknown = _normalize_point_item({"payload": {"chunk_id": "x2"}}, index=2)

    grouped = _group_points_by_document([name_only, unknown])  # type: ignore[list-item]
    keys = [group["document_key"] for group in grouped]

    assert "Fallback Name" in keys
    assert "unknown_document" in keys


def test_normalize_exposes_separate_metadata_and_parsed_text():
    point = {"payload": _sample_payload(top_level_document_id="None")}
    normalized = _normalize_point_item(point, index=1)

    assert normalized is not None
    assert normalized["parsed_text"] == "Social Content Platform"
    assert (
        normalized["node_metadata"]["document_id"]
        == "46882477-afe3-4ec9-9790-250987a857ed"
    )
    assert normalized["top_level_metadata"]["document_id"] is None
    assert "_node_content" not in normalized["top_level_metadata"]


def test_normalize_parsed_text_falls_back_to_text_resource():
    payload = _sample_payload(top_level_document_id="None")
    node = json.loads(payload["_node_content"])
    node.pop("text", None)
    node["text_resource"] = {"text": "Text from text_resource"}
    payload["_node_content"] = json.dumps(node)

    normalized = _normalize_point_item({"payload": payload}, index=1)

    assert normalized is not None
    assert normalized["parsed_text"] == "Text from text_resource"


def test_normalize_invalid_node_content_returns_empty_detail_sections():
    payload = _sample_payload(top_level_document_id="None")
    payload["_node_content"] = "{not valid json"
    normalized = _normalize_point_item({"payload": payload}, index=1)

    assert normalized is not None
    assert normalized["parsed_text"] is None
    assert normalized["node_metadata"] == {}
