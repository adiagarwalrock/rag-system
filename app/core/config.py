from pathlib import Path

from pydantic import AliasChoices, Field
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

    # OpenAI / LlamaIndex
    AI_API_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            "AI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        ),
    )
    HF_API_TOKEN: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "HF_API_TOKEN",
            "HF_TOKEN",
            "HUGGINGFACE_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
        ),
    )
    LLM_MODEL: str = "gpt-5.5"
    QUERY_EXPANSION_MODEL: str = "gpt-5.4-mini"
    SESSION_SUMMARY_MODEL: str = "gpt-5.4-mini"
    OPENAI_USE_RESPONSES: bool = True
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_OUTPUT_DIMENSION: int | None = None
    RESPONSE_INPUT_BUDGET_RATIO: float = 0.8
    RESPONSE_MAX_OUTPUT_TOKENS: int = 6000
    # max wait for main answer synthesis; gpt-5.2 reasoning_effort=high
    # on cross-document questions can take 2+ minutes
    RESPONSE_SYNTHESIS_TIMEOUT_SECONDS: int = 240
    RESPONSE_PROMPT_CACHE_KEY: str = "vectera:grounded-answer:v2"
    RESPONSE_PROMPT_CACHE_RETENTION: str = "24h"
    RESPONSE_USER_TAG: str = "developer"
    RESPONSE_SAFETY_IDENTIFIER_PREFIX: str = "vectera-client"
    TOKEN_BUDGET_ENCODING: str = "o200k_base"
    LLM_CONTEXT_WINDOW_TOKENS: int = 200000
    CHAT_SUMMARY_MAX_OUTPUT_TOKENS: int = 300
    SESSION_SUMMARY_TIMEOUT_SECONDS: int = 15

    # Retrieval reranking
    ENABLE_CROSS_ENCODER_RERANKING: bool = True
    CROSS_ENCODER_RERANK_MODEL: str = "BAAI/bge-reranker-base"
    CROSS_ENCODER_RERANK_FALLBACK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_RERANK_DEVICE: str | None = None
    CROSS_ENCODER_RERANK_TRUST_REMOTE_CODE: bool = False

    COLLECTION_NAME: str = "rag_collection_oai_parser_extractor"
    CHAT_HISTORY_COLLECTION_NAME: str = "chat_history_v1"
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
    LLM_ARTIFACT_ENRICHMENT_MAX_PAGES: int = 3
    ENABLE_MULTIMODAL_CAPTIONING: bool = True
    LLM_CAPTION_MAX_PAGES: int = 3
    LLM_CAPTION_MAX_ARTIFACTS_PER_PAGE: int = 3
    LLM_CAPTION_TIMEOUT_SECONDS: int = 25
    LLM_SCREENSHOT_TIMEOUT_SECONDS: int = 45
    LLM_SCREENSHOT_MAX_WORKERS: int = 2
    LLM_SCREENSHOT_RETRIES: int = 2

    # Non-layout parsing strategy (semantic splitter only)
    SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE: int = 95
    SEMANTIC_SPLITTER_BUFFER_SIZE: int = 1

    # External document parsers (Reducto / LlamaParse)
    # Priority order: Reducto → LlamaParse → Layout-aware PDF → Legacy
    # Each level is attempted only if its key is set; failure falls through to the next.
    ENABLE_EXTERNAL_PARSER: bool = True
    EXTERNAL_PARSER_VERSION: str = "1.0.0"
    REDUCTO_API_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices("REDUCTO_API_KEY"),
    )
    LLAMAPARSE_API_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLAMA_CLOUD_API_KEY", "LLAMAPARSE_API_KEY"),
    )

    # Reasoning enrichment (artifact-first grounded analysis)
    ENABLE_LLM_REASONING_ENRICHMENT: bool = True
    REASONING_MAX_PAGES: int = 5
    REASONING_MAX_ARTIFACTS_PER_PAGE: int = 4
    REASONING_MAX_OUTPUT_TOKENS: int = 750
    REASONING_TIMEOUT_SECONDS: int = 30

    # Use a non-reasoning model for reasoning enrichment — gpt-5.2 burns hidden
    # chain-of-thought tokens against max_output_tokens, starving the JSON output.
    REASONING_MODEL: str | None = "gpt-5.4-mini"

    # OpenAI reasoning summary verbosity: "auto", "concise", "detailed", or None to disable.
    # Only applied when OPENAI_USE_RESPONSES=True and a reasoning_effort is set.
    # Must be explicitly opted in — OpenAI does not return reasoning summaries unless reasoning.summary is set.
    # gpt-5.5 requires reasoning_effort="high" for populated summaries; "medium" returns empty summary.
    REASONING_SUMMARY: str | None = "auto"

    # Agentic RAG (LangGraph pipeline — query path only, ingestion unchanged)
    ENABLE_AGENTIC_RAG: bool = False
    # vector_retrieval_node increments after each pass; guard fires at >= this value
    AGENTIC_MAX_ITERATIONS: int = 5
    # defaults to QUERY_EXPANSION_MODEL when None
    AGENTIC_EVIDENCE_EVALUATOR_MODEL: str | None = None
    # defaults to QUERY_EXPANSION_MODEL when None
    AGENTIC_PLANNER_MODEL: str | None = None

    # Background ingestion
    INGESTION_MAX_WORKERS: int = 5
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
    def ai_api_key(self) -> str:
        return self._normalize_secret(self.AI_API_KEY)

    @property
    def hf_api_token(self) -> str:
        return self._normalize_secret(self.HF_API_TOKEN)

    @property
    def is_openai_api_key_placeholder(self) -> bool:
        key = self.ai_api_key.lower()
        placeholders = {
            "your_openai_api_key_here",
            "your_api_key_here",
            "your_api_key",
            "your_ai_api_key",
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


def validate_runtime_settings() -> None:
    """Validate mandatory runtime configuration before serving requests."""
    if settings.is_openai_api_key_placeholder:
        raise RuntimeError("AI_API_KEY must be set to a valid key before startup.")
