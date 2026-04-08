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

    # OpenAI / LlamaIndex
    OPENAI_API_KEY: str | None = None

    COLLECTION_NAME: str = "rag_collection_a"
    VECTOR_DIMENSIONS: int = 1536

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
