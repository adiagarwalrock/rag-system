from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes_clients import router
from app.db.base import Base
from app.db.snowflake import get_db
from app.db.models import Client, Document


def _session():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,
    )
    Base.metadata.create_all(bind=engine)
    return engine, SessionLocal()


def _client(db_session) -> TestClient:
    app = FastAPI()
    app.dependency_overrides[get_db] = lambda: db_session
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def _assert_numeric_counts(body: dict, expected: int = 0) -> None:
    for field in (
        "document_count",
        "query_count",
        "session_count",
        "memory_point_count",
    ):
        assert body[field] == expected
        assert isinstance(body[field], int)


def test_create_client_returns_zero_counts():
    engine, db_session = _session()
    try:
        response = _client(db_session).post(
            "/",
            json={"name": "Parser Lab", "description": "Parser comparison workspace"},
        )
    finally:
        db_session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Parser Lab"
    assert body["description"] == "Parser comparison workspace"
    _assert_numeric_counts(body)


def test_update_client_returns_numeric_counts():
    engine, db_session = _session()
    client = Client(
        id="client-1",
        name="Acme Co",
        description="Test client",
        is_active=True,
    )
    document = Document(
        id="doc-1",
        client_id=client.id,
        name="policy_v1.pdf",
        file_type=".pdf",
        storage_path="/tmp/policy_v1.pdf",
        checksum="seed-checksum",
        status="indexed",
    )
    db_session.add_all([client, document])
    db_session.commit()

    try:
        response = _client(db_session).patch(
            f"/{client.id}",
            json={"description": "Updated workspace"},
        )
    finally:
        db_session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == client.id
    assert body["description"] == "Updated workspace"
    assert body["document_count"] == 1
    for field in ("query_count", "session_count", "memory_point_count"):
        assert body[field] == 0
        assert isinstance(body[field], int)
