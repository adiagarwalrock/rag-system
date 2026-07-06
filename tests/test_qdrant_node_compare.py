import json
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import qdrant_document_compare
from app.api import routes_qdrant


@dataclass
class FakeRecord:
    id: str
    payload: dict[str, Any]
    vector: Any = None


class FakeQdrantClient:
    def __init__(self, records: list[FakeRecord]):
        self.records = records
        self.last_scroll_filter = None
        self.scroll_filters = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name == "docs"

    def scroll(self, **kwargs):
        self.last_scroll_filter = kwargs.get("scroll_filter")
        self.scroll_filters.append(self.last_scroll_filter)
        limit = kwargs.get("limit") or len(self.records)
        offset = kwargs.get("offset")
        start = int(offset or 0)
        end = start + limit
        next_offset = end if end < len(self.records) else None
        return self.records[start:end], next_offset

    def retrieve(self, **kwargs):
        ids = {str(item) for item in kwargs.get("ids", [])}
        return [record for record in self.records if record.id in ids]


def _client(monkeypatch, fake_qdrant: FakeQdrantClient) -> TestClient:
    app = FastAPI()
    app.include_router(routes_qdrant.router)
    monkeypatch.setattr(
        routes_qdrant.vector_store_manager,
        "get_qdrant_client",
        lambda: fake_qdrant,
    )
    return TestClient(app, raise_server_exceptions=False)


def _payload(
    *,
    node_id: str,
    document_id: str = "nested-doc",
    document_name: str = "Digital Realty Deck.pdf",
    client_id: str = "client-1",
    parser_name: str = "reducto",
    parser_version: str = "1.0",
    chunk_type: str = "body_text",
    page_num: int = 7,
    text: str = "Node body text",
) -> dict[str, Any]:
    metadata = {
        "document_id": document_id,
        "document_name": document_name,
        "client_id": client_id,
        "parser_name": parser_name,
        "parser_version": parser_version,
        "chunk_id": f"chunk-{node_id}",
        "chunk_type": chunk_type,
        "page_num": page_num,
        "page_nums": [page_num],
        "section_path": "Overview",
        "citation_label": f"{document_name} - p.{page_num}",
    }
    return {
        "document_id": "None",
        "document_name": document_name,
        "client_id": client_id,
        "parser_name": parser_name,
        "parser_version": parser_version,
        "chunk_type": chunk_type,
        "page_num": page_num,
        "_node_content": json.dumps(
            {"id_": node_id, "metadata": metadata, "text": text}
        ),
    }


def test_serialize_node_prefers_nested_metadata_and_text():
    record = FakeRecord(id="node-1", payload=_payload(node_id="node-1"))

    node = routes_qdrant._serialize_node(
        "docs",
        record,
        include_raw_payload=True,
    )

    assert node["id"] == "node-1"
    assert node["document_id"] == "nested-doc"
    assert node["document_name"] == "Digital Realty Deck.pdf"
    assert node["parser_name"] == "reducto"
    assert node["chunk_type"] == "body_text"
    assert node["page_num"] == 7
    assert node["page_nums"] == [7]
    assert node["text"] == "Node body text"
    assert node["text_preview"] == "Node body text"
    assert node["node_metadata"]["chunk_id"] == "chunk-node-1"
    assert "_node_content" not in node["top_level_metadata"]
    assert "raw_payload" in node


def test_serialize_node_uses_text_resource_fallback():
    payload = _payload(node_id="node-1")
    node_content = json.loads(payload["_node_content"])
    node_content.pop("text")
    node_content["text_resource"] = {"text": "Text from resource"}
    payload["_node_content"] = json.dumps(node_content)

    node = routes_qdrant._serialize_node(
        "docs",
        FakeRecord(id="node-1", payload=payload),
        include_raw_payload=False,
    )

    assert node["text"] == "Text from resource"
    assert node["text_length"] == len("Text from resource")
    assert "raw_payload" not in node


def test_list_nodes_filters_by_document_title_and_returns_facets(monkeypatch):
    records = [
        FakeRecord(id="node-1", payload=_payload(node_id="node-1")),
        FakeRecord(
            id="node-2",
            payload=_payload(
                node_id="node-2",
                document_name="Other Deck.pdf",
                parser_name="llamaparse",
                text="Other node",
            ),
        ),
    ]
    response = _client(monkeypatch, FakeQdrantClient(records)).get(
        "/collections/docs/nodes",
        params={"document_title": "Digital Realty", "limit": 10},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["nodes"][0]["id"] == "node-1"
    assert body["facets"]["documents"][0]["title"] == "Digital Realty Deck.pdf"
    assert body["facets"]["parsers"] == [{"name": "reducto", "count": 1}]


def test_compare_nodes_preserves_requested_order(monkeypatch):
    records = [
        FakeRecord(id="node-1", payload=_payload(node_id="node-1")),
        FakeRecord(id="node-2", payload=_payload(node_id="node-2", text="second")),
    ]
    response = _client(monkeypatch, FakeQdrantClient(records)).get(
        "/collections/docs/nodes/compare",
        params=[("point_ids", "node-2"), ("point_ids", "node-1")],
    )

    assert response.status_code == 200
    assert [node["id"] for node in response.json()["nodes"]] == ["node-2", "node-1"]


def test_compare_nodes_requires_two_to_four_nodes(monkeypatch):
    response = _client(monkeypatch, FakeQdrantClient([])).get(
        "/collections/docs/nodes/compare",
        params={"point_ids": "node-1"},
    )

    assert response.status_code == 422


def test_compare_nodes_reports_missing_points(monkeypatch):
    records = [FakeRecord(id="node-1", payload=_payload(node_id="node-1"))]
    response = _client(monkeypatch, FakeQdrantClient(records)).get(
        "/collections/docs/nodes/compare",
        params=[("point_ids", "node-1"), ("point_ids", "missing")],
    )

    assert response.status_code == 404
    assert response.json()["detail"]["missing"] == ["missing"]


def test_list_documents_groups_nodes_by_workspace_document_and_parser(monkeypatch):
    fake_qdrant = FakeQdrantClient(
        [
            FakeRecord(
                id="node-2",
                payload=_payload(
                    node_id="node-2",
                    document_id="doc-a",
                    document_name="Parser Diff.pdf",
                    parser_name="reducto",
                    page_num=2,
                    text="Second page",
                ),
            ),
            FakeRecord(
                id="node-1",
                payload=_payload(
                    node_id="node-1",
                    document_id="doc-a",
                    document_name="Parser Diff.pdf",
                    parser_name="reducto",
                    page_num=1,
                    text="First page",
                ),
            ),
            FakeRecord(
                id="node-3",
                payload=_payload(
                    node_id="node-3",
                    document_id="doc-a",
                    document_name="Parser Diff.pdf",
                    client_id="client-2",
                    parser_name="reducto",
                    page_num=1,
                    text="Client two copy",
                ),
            ),
            FakeRecord(
                id="node-4",
                payload=_payload(
                    node_id="node-4",
                    document_id="doc-a",
                    document_name="Parser Diff.pdf",
                    parser_name="llamaparse",
                    page_num=1,
                    text="Different parser",
                ),
            ),
        ]
    )

    response = _client(monkeypatch, fake_qdrant).get(
        "/collections/docs/documents",
        params={"document_title": "Parser Diff", "limit": 10},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    reducto_client_one = next(
        item
        for item in body["documents"]
        if item["client_id"] == "client-1" and item["parser_name"] == "reducto"
    )
    assert reducto_client_one["node_count"] == 2
    assert reducto_client_one["page_count"] == 2
    assert reducto_client_one["preview"] == "First page Second page"
    assert {item["name"] for item in body["facets"]["parsers"]} == {
        "llamaparse",
        "reducto",
    }
    assert fake_qdrant.last_scroll_filter is None


def test_compare_documents_reconstructs_markdown_and_preserves_order(monkeypatch):
    records = [
        FakeRecord(
            id="a-2",
            payload=_payload(
                node_id="a-2",
                document_id="doc-a",
                document_name="A.pdf",
                page_num=2,
                text="## Page two",
            ),
        ),
        FakeRecord(
            id="a-1",
            payload=_payload(
                node_id="a-1",
                document_id="doc-a",
                document_name="A.pdf",
                page_num=1,
                text="# Page one",
            ),
        ),
        FakeRecord(
            id="b-1",
            payload=_payload(
                node_id="b-1",
                document_id="doc-b",
                document_name="B.pdf",
                parser_name="llamaparse",
                page_num=1,
                text="# Other document",
            ),
        ),
    ]
    client = _client(monkeypatch, FakeQdrantClient(records))
    list_response = client.get("/collections/docs/documents", params={"limit": 10})
    keys_by_name = {
        item["document_name"]: item["document_key"]
        for item in list_response.json()["documents"]
    }

    response = client.get(
        "/collections/docs/documents/compare",
        params=[
            ("document_keys", keys_by_name["B.pdf"]),
            ("document_keys", keys_by_name["A.pdf"]),
        ],
    )

    assert response.status_code == 200
    documents = response.json()["documents"]
    assert [document["document_name"] for document in documents] == ["B.pdf", "A.pdf"]
    assert documents[1]["markdown"] == "# Page one\n\n## Page two"
    assert [node["id"] for node in documents[1]["source_nodes"]] == ["a-1", "a-2"]
    assert documents[1]["metadata_summary"]["pages"] == [1, 2]


def test_compare_documents_does_not_filter_qdrant_by_parser_version(monkeypatch):
    records = [
        FakeRecord(
            id="v1",
            payload=_payload(
                node_id="v1",
                document_id="doc-a",
                document_name="A.pdf",
                parser_version="1.0.0",
                text="Version one",
            ),
        ),
        FakeRecord(
            id="v2",
            payload=_payload(
                node_id="v2",
                document_id="doc-a",
                document_name="A.pdf",
                parser_version="2.0.0",
                text="Version two",
            ),
        ),
        FakeRecord(
            id="other",
            payload=_payload(
                node_id="other",
                document_id="doc-b",
                document_name="B.pdf",
                text="Other document",
            ),
        ),
    ]
    fake_qdrant = FakeQdrantClient(records)
    client = _client(monkeypatch, fake_qdrant)
    list_response = client.get("/collections/docs/documents", params={"limit": 10})
    keys_by_version = {
        item["parser_version"]: item["document_key"]
        for item in list_response.json()["documents"]
        if item["document_name"] == "A.pdf"
    }
    other_key = next(
        item["document_key"]
        for item in list_response.json()["documents"]
        if item["document_name"] == "B.pdf"
    )

    response = client.get(
        "/collections/docs/documents/compare",
        params=[
            ("document_keys", keys_by_version["2.0.0"]),
            ("document_keys", other_key),
        ],
    )

    assert response.status_code == 200
    assert response.json()["documents"][0]["markdown"] == "Version two"
    compare_filter = fake_qdrant.scroll_filters[-2]
    filtered_keys = {
        getattr(condition, "key", None)
        for condition in getattr(compare_filter, "must", []) or []
    }
    assert "parser_version" not in filtered_keys


def test_compare_documents_requires_two_to_four_documents(monkeypatch):
    response = _client(monkeypatch, FakeQdrantClient([])).get(
        "/collections/docs/documents/compare",
        params={"document_keys": "one"},
    )

    assert response.status_code == 422


def test_compare_documents_rejects_invalid_key(monkeypatch):
    valid_key = qdrant_document_compare._document_key_for_node(
        routes_qdrant._serialize_node(
            "docs",
            FakeRecord(id="node-1", payload=_payload(node_id="node-1")),
            include_raw_payload=False,
        )
    )

    response = _client(monkeypatch, FakeQdrantClient([])).get(
        "/collections/docs/documents/compare",
        params=[("document_keys", "not-valid-base64"), ("document_keys", valid_key)],
    )

    assert response.status_code == 422


def test_compare_documents_reports_missing_keys(monkeypatch):
    missing_key = qdrant_document_compare._document_key_for_node(
        routes_qdrant._serialize_node(
            "docs",
            FakeRecord(
                id="missing",
                payload=_payload(
                    node_id="missing",
                    document_id="missing-doc",
                    document_name="Missing.pdf",
                ),
            ),
            include_raw_payload=False,
        )
    )
    present_key = qdrant_document_compare._document_key_for_node(
        routes_qdrant._serialize_node(
            "docs",
            FakeRecord(id="node-1", payload=_payload(node_id="node-1")),
            include_raw_payload=False,
        )
    )

    response = _client(
        monkeypatch,
        FakeQdrantClient([FakeRecord(id="node-1", payload=_payload(node_id="node-1"))]),
    ).get(
        "/collections/docs/documents/compare",
        params=[("document_keys", present_key), ("document_keys", missing_key)],
    )

    assert response.status_code == 404
    assert response.json()["detail"]["missing"] == [missing_key]
