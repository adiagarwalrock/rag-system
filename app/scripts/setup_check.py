"""
CLI setup check for Snowflake, Qdrant, and LLM configuration.

Run:
    uv run python -m app.scripts.setup_check
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from urllib.parse import quote_plus

import requests
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import settings
from app.core.logging_config import configure_logging

SPARSE_VECTOR_NAME = "text-sparse-new"

warnings.filterwarnings(
    "ignore",
    message="Api key is used with an insecure connection.",
    category=UserWarning,
)

configure_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.FileHandler):
        if handler not in logger.handlers:
            logger.addHandler(handler)


def emit(message: str, *, level: int = logging.INFO) -> None:
    print(message)
    logger.log(level, message)


def check_snowflake() -> bool:
    required_values = {
        "SNOWFLAKE_ACCOUNT": settings.SNOWFLAKE_ACCOUNT,
        "SNOWFLAKE_USER": settings.SNOWFLAKE_USER,
        "SNOWFLAKE_PASSWORD": settings.SNOWFLAKE_PASSWORD,
        "SNOWFLAKE_DATABASE": settings.SNOWFLAKE_DATABASE,
        "SNOWFLAKE_SCHEMA": settings.SNOWFLAKE_SCHEMA,
        "SNOWFLAKE_WAREHOUSE": settings.SNOWFLAKE_WAREHOUSE,
        "SNOWFLAKE_ROLE": settings.SNOWFLAKE_ROLE,
    }
    missing = [name for name, value in required_values.items() if not value]
    if missing:
        emit("[FAIL] Snowflake: missing required settings:", level=logging.ERROR)
        emit(f"       {', '.join(missing)}", level=logging.ERROR)
        return False

    conn_str = (
        f"snowflake://{settings.SNOWFLAKE_USER}:{quote_plus(settings.SNOWFLAKE_PASSWORD or '')}"
        f"@{settings.SNOWFLAKE_ACCOUNT}"
        f"/{settings.SNOWFLAKE_DATABASE}"
        f"/{settings.SNOWFLAKE_SCHEMA}"
        f"?warehouse={settings.SNOWFLAKE_WAREHOUSE}"
        f"&role={settings.SNOWFLAKE_ROLE}"
    )
    engine = create_engine(conn_str, echo=False)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        emit("[PASS] Snowflake: connection successful")
        return True
    except Exception as exc:
        emit(f"[FAIL] Snowflake: connection failed ({exc})", level=logging.ERROR)
        return False
    finally:
        engine.dispose()


def check_qdrant() -> bool:
    try:
        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        client.get_collections()
        emit(f"[PASS] Qdrant: reachable at {settings.QDRANT_URL}")

        if client.collection_exists(settings.COLLECTION_NAME):
            info = client.get_collection(settings.COLLECTION_NAME)
            sparse_vectors = info.config.params.sparse_vectors or {}
            if SPARSE_VECTOR_NAME in sparse_vectors:
                emit(
                    f"[PASS] Qdrant: collection '{settings.COLLECTION_NAME}' has hybrid sparse vectors"
                )
            else:
                emit(
                    f"[WARN] Qdrant: collection '{settings.COLLECTION_NAME}' exists but is dense-only",
                    level=logging.WARNING,
                )
        else:
            emit(
                f"[WARN] Qdrant: collection '{settings.COLLECTION_NAME}' not found yet (created on first index)",
                level=logging.WARNING,
            )

        return True
    except Exception as exc:
        emit(f"[FAIL] Qdrant: connection failed ({exc})", level=logging.ERROR)
        return False


def check_llm() -> bool:
    api_key = settings.ai_api_key
    if settings.is_openai_api_key_placeholder:
        emit("[FAIL] LLM: AI_API_KEY is missing or placeholder", level=logging.ERROR)
        return False

    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
    except requests.RequestException as exc:
        emit(f"[FAIL] LLM: OpenAI API unreachable ({exc})", level=logging.ERROR)
        return False

    if response.status_code == 200:
        emit("[PASS] LLM: AI_API_KEY accepted by OpenAI API")
        return True

    if response.status_code in {401, 403}:
        emit("[FAIL] LLM: AI_API_KEY rejected by OpenAI API", level=logging.ERROR)
        return False

    emit(
        f"[FAIL] LLM: OpenAI API returned status {response.status_code}",
        level=logging.ERROR,
    )
    return False


def main() -> None:
    emit("RAG-System setup check")
    emit("=" * 50)

    snowflake_ok = check_snowflake()
    qdrant_ok = check_qdrant()
    llm_ok = check_llm()

    emit("=" * 50)
    passed = int(snowflake_ok) + int(qdrant_ok) + int(llm_ok)
    emit(f"Checks passed: {passed}/3")

    if passed == 3:
        emit("[READY] All required integrations are configured correctly.")
        sys.exit(0)

    emit("[NOT READY] Fix failed checks and run again.", level=logging.ERROR)
    sys.exit(1)


if __name__ == "__main__":
    main()
