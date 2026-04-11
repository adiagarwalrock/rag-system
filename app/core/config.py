from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Backend"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "supersecretkey_please_change"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    AUTH_ENABLED: bool = True

    # Database
    SNOWFLAKE_ACCOUNT: str | None = None
    SNOWFLAKE_USER: str | None = None
    SNOWFLAKE_PASSWORD: str | None = None
    SNOWFLAKE_DATABASE: str | None = None
    SNOWFLAKE_SCHEMA: str | None = None
    SNOWFLAKE_WAREHOUSE: str | None = None
    SNOWFLAKE_ROLE: str | None = None

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None

    # Local file storage
    RAW_DATA_DIR: str = str(Path(__file__).resolve().parents[2] / "data" / "raw")

    # Google GenAI / LlamaIndex
    GOOGLE_API_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    LLM_MODEL: str = "gemini-3-flash-preview"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_OUTPUT_DIMENSION: int | None = None

    COLLECTION_NAME: str = "rag_collection"
    VECTOR_DIMENSIONS: int = 3072

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @staticmethod
    def _normalize_secret(value: str | None) -> str:
        key = (value or "").strip()
        return key.strip("'\"").strip()

    @property
    def google_api_key(self) -> str:
        return self._normalize_secret(self.GOOGLE_API_KEY)

    @property
    def is_google_api_key_placeholder(self) -> bool:
        key = self.google_api_key.lower()
        placeholders = {
            "your_google_api_key_here",
            "your_gemini_api_key_here",
            "your_api_key_here",
            "your_api_key",
        }
        return not key or key in placeholders or key.startswith("your_")

    @property
    def effective_vector_dimensions(self) -> int:
        return self.EMBEDDING_OUTPUT_DIMENSION or self.VECTOR_DIMENSIONS

    @property
    def VECTOR_COLLECTION(self) -> str:
        """Backward-compatible alias for tests/imports using legacy naming."""
        return self.COLLECTION_NAME


settings = Settings()
