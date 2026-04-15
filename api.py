from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_auth import router as auth_router
from app.api.routes_clients import router as clients_router
from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.api.routes_query import router as query_router
from app.core.ai_provider import initialize_ai_provider
from app.core.config import settings, validate_runtime_settings
from app.db.base import Base

# Import all models so Base.metadata knows about every table
from app.db.models import *  # noqa: F401, F403
from app.db.snowflake import engine

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)


@app.on_event("startup")
def _validate_runtime_config() -> None:
    validate_runtime_settings()
    initialize_ai_provider()


# Auto-create all tables on startup
Base.metadata.create_all(bind=engine)

# Register routes
app.include_router(auth_router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(
    clients_router, prefix=f"{settings.API_V1_STR}/clients", tags=["clients"]
)
app.include_router(
    documents_router, prefix=f"{settings.API_V1_STR}/documents", tags=["documents"]
)
app.include_router(query_router, prefix=f"{settings.API_V1_STR}/query", tags=["query"])
app.include_router(
    health_router, prefix=f"{settings.API_V1_STR}/health", tags=["health"]
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
