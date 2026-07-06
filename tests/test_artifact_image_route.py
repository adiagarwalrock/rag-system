from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes_artifacts import router
from app.core.config import settings


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def test_artifact_image_route_serves_parsed_image(monkeypatch, tmp_path):
    root = tmp_path / "parsed"
    image = root / "doc-1" / "screenshots" / "page_1.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake image")
    monkeypatch.setattr(settings, "PARSED_ARTIFACTS_DIR", str(root))

    response = _client().get("/image", params={"path": "doc-1/screenshots/page_1.png"})

    assert response.status_code == 200
    assert response.content == b"fake image"


def test_artifact_image_route_rejects_path_traversal(monkeypatch, tmp_path):
    root = tmp_path / "parsed"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"fake image")
    monkeypatch.setattr(settings, "PARSED_ARTIFACTS_DIR", str(root))

    response = _client().get("/image", params={"path": "../outside.png"})

    assert response.status_code == 400


def test_artifact_image_route_rejects_non_images(monkeypatch, tmp_path):
    root = tmp_path / "parsed"
    text = root / "doc-1" / "note.txt"
    text.parent.mkdir(parents=True)
    text.write_text("not an image", encoding="utf-8")
    monkeypatch.setattr(settings, "PARSED_ARTIFACTS_DIR", str(root))

    response = _client().get("/image", params={"path": str(Path("doc-1") / "note.txt")})

    assert response.status_code == 400
