from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Backend"
    API_V1_STR: str = "/api/v1"

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
    PARSED_ARTIFACTS_DIR: str = str(
        Path(__file__).resolve().parents[2] / "artifacts" / "parsed"
    )

    # AI provider keys and models
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    AI_API_KEY: str | None = None
    # LLM_MODEL: str = "gemini-2.5-flash"
    # QUERY_EXPANSION_MODEL: str = "gemini-2.5-flash"
    # SESSION_SUMMARY_MODEL: str = "gemini-2.5-flash"

    LLM_MODEL: str = "gpt-5.2"
    QUERY_EXPANSION_MODEL: str = "gpt-5.4-mini"
    SESSION_SUMMARY_MODEL: str = "gpt-5.4-mini"

    OPENAI_USE_RESPONSES: bool = True
    # EMBEDDING_MODEL: str = "gemini-embedding-2-preview"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_OUTPUT_DIMENSION: int | None = None
    RESPONSE_INPUT_BUDGET_RATIO: float = 0.8
    RESPONSE_MAX_OUTPUT_TOKENS: int = 1200
    RESPONSE_PROMPT_CACHE_KEY: str = "rag:grounded-answer:v2"
    RESPONSE_PROMPT_CACHE_RETENTION: str = "24h"
    RESPONSE_USER_TAG: str = "developer"
    RESPONSE_SAFETY_IDENTIFIER_PREFIX: str = "rag-client"
    TOKEN_BUDGET_ENCODING: str = "o200k_base"
    LLM_CONTEXT_WINDOW_TOKENS: int = 200000
    CHAT_SUMMARY_MAX_OUTPUT_TOKENS: int = 300
    SESSION_SUMMARY_TIMEOUT_SECONDS: int = 15

    COLLECTION_NAME: str = "rag_collection_oai_v2"
    CHAT_HISTORY_COLLECTION_NAME: str = "chat_history"
    VECTOR_DIMENSIONS: int = 3072

    # Layout-aware PDF ingestion
    ENABLE_LAYOUT_AWARE_PDF: bool = True
    STRICT_LAYOUT_AWARE_PDF_FAILURE: bool = False
    ENABLE_OCR_FALLBACK: bool = True
    ENABLE_MULTIPAGE_TABLE_MERGE: bool = True
    PDF_LAYOUT_PARSER_VERSION: str = "3.0.0"
    BODY_TEXT_CHUNK_MAX_TOKENS: int = 600
    BODY_TEXT_CHUNK_OVERLAP_TOKENS: int = 40
    TABLE_CHUNK_ROW_THRESHOLD: int = 6
    ENABLE_PYMUPDF_LAYOUT: bool = True
    ENABLE_PDF_REPAIR_PREPASS: bool = True
    SUPPRESS_MUPDF_STDERR: bool = True
    ENABLE_LLM_ARTIFACT_ENRICHMENT: bool = True
    LLM_ARTIFACT_ENRICHMENT_MAX_PAGES: int = 8
    ENABLE_MULTIMODAL_CAPTIONING: bool = True
    LLM_CAPTION_MAX_PAGES: int = 8
    LLM_CAPTION_MAX_ARTIFACTS_PER_PAGE: int = 5
    LLM_CAPTION_TIMEOUT_SECONDS: int = 25
    LLM_SCREENSHOT_TIMEOUT_SECONDS: int = 45
    LLM_SCREENSHOT_MAX_WORKERS: int = 2
    LLM_SCREENSHOT_RETRIES: int = 2

    # Non-layout parsing strategy (semantic splitter only)
    SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE: int = 95
    SEMANTIC_SPLITTER_BUFFER_SIZE: int = 1

    # Reasoning enrichment (artifact-first grounded analysis)
    ENABLE_LLM_REASONING_ENRICHMENT: bool = True
    REASONING_MAX_PAGES: int = 12
    REASONING_MAX_ARTIFACTS_PER_PAGE: int = 4
    REASONING_MAX_OUTPUT_TOKENS: int = 700
    REASONING_TIMEOUT_SECONDS: int = 30
    REASONING_MODEL: str | None = None  # defaults to LLM_MODEL if None

    # Background ingestion
    INGESTION_MAX_WORKERS: int = 2
    INGESTION_QUEUE_MAX_SIZE: int = 128

    # UI responsiveness
    UI_POLL_INTERVAL_SECONDS: int = 3
    CHAT_SESSION_RECENT_TURNS: int = 8
    CHAT_CROSS_SESSION_TOP_K: int = 4

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
    def openai_api_key(self) -> str:
        return self._normalize_secret(self.OPENAI_API_KEY)

    @property
    def gemini_api_key(self) -> str:
        return self._normalize_secret(self.GEMINI_API_KEY)

    @property
    def google_api_key(self) -> str:
        return self._normalize_secret(self.GOOGLE_API_KEY)

    @property
    def fallback_ai_api_key(self) -> str:
        return self._normalize_secret(self.AI_API_KEY)

    @property
    def ai_provider(self) -> str:
        if self.gemini_api_key or self.google_api_key:
            return "gemini"
        if self.openai_api_key:
            return "openai"
        if "gemini" in self.LLM_MODEL.lower():
            return "gemini"
        return "openai"

    @property
    def ai_api_key(self) -> str:
        if self.ai_provider == "gemini":
            return (
                self.gemini_api_key or self.google_api_key or self.fallback_ai_api_key
            )
        return self.openai_api_key or self.fallback_ai_api_key

    @property
    def is_ai_api_key_placeholder(self) -> bool:
        key = self.ai_api_key.lower()
        placeholders = {
            "your_openai_api_key_here",
            "your_api_key_here",
            "your_api_key",
            "your_ai_api_key",
            "your_gemini_api_key_here",
            "your_google_api_key_here",
        }
        return not key or key in placeholders or key.startswith("your_")

    @property
    def is_openai_api_key_placeholder(self) -> bool:
        """Backward-compatible alias used by existing callers/tests."""
        return self.is_ai_api_key_placeholder

    @property
    def effective_vector_dimensions(self) -> int:
        return self.EMBEDDING_OUTPUT_DIMENSION or self.VECTOR_DIMENSIONS

    @property
    def VECTOR_COLLECTION(self) -> str:
        """Backward-compatible alias for tests/imports using legacy naming."""
        return self.COLLECTION_NAME


settings = Settings()


def validate_runtime_settings() -> None:
    """Validate mandatory runtime configuration before serving requests."""
    if settings.is_ai_api_key_placeholder:
        raise RuntimeError("AI_API_KEY must be set to a valid key before startup.")
