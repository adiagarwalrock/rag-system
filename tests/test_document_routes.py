from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes_documents import router
from app.db.base import Base
from app.db.models import Client, Document, IngestionJob, VectorNodeRegistry
from app.db.snowflake import get_db


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


def test_list_documents_returns_parser_and_vector_counts():
    engine, db_session = _session()
    client = Client(
        id="client-1",
        name="Parser Lab",
        description="Test workspace",
        is_active=True,
    )
    document = Document(
        id="doc-1",
        client_id=client.id,
        name="deck.pdf",
        file_type=".pdf",
        storage_path="/tmp/deck.pdf",
        checksum="checksum",
        status="indexed",
    )
    db_session.add_all([client, document])
    db_session.flush()
    db_session.add_all(
        [
            IngestionJob(
                id="old-job",
                client_id=client.id,
                document_id=document.id,
                status="failed",
                started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                parser_name="rag_ingestion_pipeline",
                filesize_bytes=10,
            ),
            IngestionJob(
                id="latest-job",
                client_id=client.id,
                document_id=document.id,
                status="completed",
                started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                parser_name="llamaparse",
                filesize_bytes=10,
            ),
            VectorNodeRegistry(
                id="vec-1",
                document_id=document.id,
                client_id=client.id,
                vector_collection="docs",
                vector_node_id="node-1",
                is_active=True,
            ),
            VectorNodeRegistry(
                id="vec-2",
                document_id=document.id,
                client_id=client.id,
                vector_collection="docs",
                vector_node_id="node-2",
                is_active=True,
            ),
            VectorNodeRegistry(
                id="vec-inactive",
                document_id=document.id,
                client_id=client.id,
                vector_collection="docs",
                vector_node_id="node-3",
                is_active=False,
            ),
        ]
    )
    db_session.commit()

    try:
        response = _client(db_session).get("/", params={"client_id": client.id})
    finally:
        db_session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()

    assert response.status_code == 200
    body = response.json()
    assert body[0]["id"] == document.id
    assert body[0]["parser_used"] == "llamaparse"
    assert body[0]["vector_point_count"] == 2
    assert isinstance(body[0]["vector_point_count"], int)
