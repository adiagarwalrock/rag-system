from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_clients import router as clients_router
from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.api.routes_query import router as query_router
from app.agents.agent_base import initialize_ai_provider
from app.core.config import settings, validate_runtime_settings
from observability.logging_config import configure_logging

# Import all models so Base.metadata knows about every table
from app.db.models import *  # noqa: F401, F403
from app.db.schema import ensure_runtime_schema
from app.db.snowflake import engine

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_runtime_settings()
    initialize_ai_provider()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)


# Auto-create all tables on startup
ensure_runtime_schema(engine)

# Register routes
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
