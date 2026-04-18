"""
File validation for uploads: type, size, and checksums.
"""

import hashlib
import os

from fastapi import HTTPException, UploadFile

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx"}
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_file_type(filename: str) -> str:
    """Validate and return the file extension.

    Args:
        filename: Name of the file being uploaded.

    Returns:
        Lower-cased validated extension string.
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    return ext


def validate_file_size(file_content: bytes) -> int:
    """Validate file size and return size in bytes.

    Args:
        file_content: Raw bytes representation of the file.

    Returns:
        Integer representing the size in bytes.
    """
    size = len(file_content)
    if size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({size / 1024 / 1024:.1f}MB) exceeds maximum ({MAX_FILE_SIZE_MB}MB)",
        )
    if size == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    return size


def compute_checksum(file_content: bytes) -> str:
    """Compute SHA-256 checksum for duplicate detection.

    Args:
        file_content: Raw bytes representation of the file.

    Returns:
        Hex-encoded SHA-256 string.
    """
    return hashlib.sha256(file_content).hexdigest()


def check_duplicate(checksum: str, db, Document) -> bool:
    """Check if a document with the same checksum already exists.

    Args:
        checksum: Target string checksum to verify.
        db: SQLAlchemy session instance.
        Document: SQLAlchemy Document model class.

    Returns:
        True if the file already exists, False otherwise.
    """
    return (
        db.query(Document.id).filter(Document.checksum == checksum).first() is not None
    )
