import pytest
from fastapi import HTTPException

from app.db.models import Document
from app.ingestion.validator import (
    MAX_FILE_SIZE_BYTES,
    check_duplicate,
    compute_checksum,
    validate_file_size,
    validate_file_type,
)


def test_validate_file_type_accepts_supported_extensions_case_insensitively():
    assert validate_file_type("Annual_Report.PDF") == ".pdf"
    assert validate_file_type("roadmap.DoCx") == ".docx"


def test_validate_file_type_rejects_unsupported_extensions():
    with pytest.raises(HTTPException) as exc:
        validate_file_type("notes.txt")

    assert exc.value.status_code == 400
    assert "Unsupported file type" in exc.value.detail


def test_validate_file_size_rejects_empty_payloads():
    with pytest.raises(HTTPException) as exc:
        validate_file_size(b"")

    assert exc.value.status_code == 400
    assert exc.value.detail == "File is empty"


def test_validate_file_size_rejects_oversized_payloads():
    with pytest.raises(HTTPException) as exc:
        validate_file_size(b"x" * (MAX_FILE_SIZE_BYTES + 1))

    assert exc.value.status_code == 400
    assert "exceeds maximum" in exc.value.detail


def test_validate_file_size_returns_size_for_valid_payloads():
    payload = b"hello world"
    assert validate_file_size(payload) == len(payload)


def test_compute_checksum_is_deterministic():
    payload = b"same input every time"
    assert compute_checksum(payload) == compute_checksum(payload)


def test_check_duplicate_returns_true_when_checksum_exists(db_session, seeded_entities):
    existing_checksum = seeded_entities["document"].checksum

    assert check_duplicate(existing_checksum, db_session, Document) is True


def test_check_duplicate_returns_false_when_checksum_is_new(db_session):
    assert check_duplicate("new-checksum", db_session, Document) is False
