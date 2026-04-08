from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.base import Base
from app.db.models import Client, Document, User


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,
    )
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture()
def seeded_entities(db_session):
    user = User(
        id="user-1",
        email="user@example.com",
        password_hash="fake-hash",
        is_active=True,
        full_name="Test User",
    )
    client = Client(
        id="client-1",
        name="Acme Co",
        description="Test client",
        created_by=user.id,
        is_active=True,
    )
    document = Document(
        id="doc-1",
        client_id=client.id,
        name="policy_v1.pdf",
        file_type=".pdf",
        storage_path="/tmp/policy_v1.pdf",
        checksum="seed-checksum",
        uploaded_by=user.id,
        status="indexed",
    )

    db_session.add_all([user, client, document])
    db_session.commit()

    return {"user": user, "client": client, "document": document}
