from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openai import OpenAI
from sqlalchemy import text

from app.core import ai_provider
from app.core.config import settings
from app.core.embedding_manager import embedding_manager
from app.core.models.llm_manager import llm_manager
from app.core.models.embedding.registry import EMBEDDING_REGISTRY_IDS
from app.core.models.llm.registry import LLM_REGISTRY_IDS
from app.db.snowflake import SessionLocal, engine
from app.indexing.vector_store import vector_store_manager


class RuntimeStatusService:
    """Read-only runtime diagnostics for the Streamlit status page."""

    def __init__(
        self,
        *,
        settings_obj: Any = settings,
        engine_obj: Any = engine,
        session_factory: Callable[[], Any] = SessionLocal,
        qdrant_client_factory: Callable[[], Any] | None = None,
        openai_client_factory: Callable[..., Any] | None = None,
        ai_initializer: Callable[[], None] = ai_provider.initialize_ai_provider,
        openai_timeout_seconds: float = 5.0,
    ) -> None:
        self.settings = settings_obj
        self.engine = engine_obj
        self.session_factory = session_factory
        self.qdrant_client_factory = (
            qdrant_client_factory or vector_store_manager.get_qdrant_client
        )
        self.openai_client_factory = openai_client_factory or OpenAI
        self.ai_initializer = ai_initializer
        self.openai_timeout_seconds = openai_timeout_seconds

    def get_status(self) -> dict[str, Any]:
        parser_status = self._parser_status()
        db_status = self._database_status()
        qdrant_status = self._qdrant_status()
        ai_status = self._ai_status()

        return {
            "overall": {
                "parsers": self._overall_status(parser_status),
                "database": db_status["status"],
                "qdrant": qdrant_status["status"],
                "ai": ai_status["status"],
            },
            "parsers": parser_status,
            "database": db_status,
            "qdrant": qdrant_status,
            "ai": ai_status,
        }

    def _parser_status(self) -> dict[str, Any]:
        external_enabled = bool(self.settings.ENABLE_EXTERNAL_PARSER)
        reducto_configured = self._has_secret(self.settings.REDUCTO_API_KEY)
        llamaparse_configured = self._has_secret(self.settings.LLAMAPARSE_API_KEY)

        parsers = [
            {
                "name": "External parser",
                "configured": external_enabled,
                "enabled": external_enabled,
                "priority": 0,
                "detail": "Controls Reducto and LlamaParse routing.",
            },
            {
                "name": "Reducto",
                "configured": reducto_configured,
                "enabled": external_enabled and reducto_configured,
                "priority": 1,
                "detail": "First external parser attempted.",
            },
            {
                "name": "LlamaParse",
                "configured": llamaparse_configured,
                "enabled": external_enabled and llamaparse_configured,
                "priority": 2,
                "detail": "Second external parser attempted.",
            },
        ]
        routing_priority = [
            parser["name"] for parser in parsers[1:] if parser["enabled"]
        ]
        routing_priority.extend(["Layout-aware PDF", "Legacy"])

        return {
            "status": "ok",
            "external_enabled": external_enabled,
            "parsers": parsers,
            "routing_priority": routing_priority,
        }

    def _database_status(self) -> dict[str, Any]:
        mode = (
            "snowflake"
            if self.settings.SNOWFLAKE_ACCOUNT and self.settings.SNOWFLAKE_USER
            else "local_sqlite"
        )
        payload = {
            "status": "ok",
            "mode": mode,
            "dialect": getattr(
                getattr(self.engine, "dialect", None), "name", "unknown"
            ),
            "target": self._database_target(mode),
            "message": "Connection check succeeded.",
        }
        try:
            with self.session_factory() as db:
                db.execute(text("SELECT 1"))
        except Exception as exc:
            payload["status"] = "error"
            payload["message"] = self._error_message(exc)
        return payload

    def _qdrant_status(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "ok",
            "url": self.settings.QDRANT_URL,
            "document_collection": self.settings.COLLECTION_NAME,
            "chat_history_collection": self.settings.CHAT_HISTORY_COLLECTION_NAME,
            "collections": [],
            "message": "Collection list loaded.",
        }
        try:
            client = self.qdrant_client_factory()
            response = client.get_collections()
            collections = getattr(response, "collections", []) or []
            rows = []
            partial_errors: list[str] = []
            for collection in collections:
                name = str(getattr(collection, "name", collection))
                point_count = None
                try:
                    point_count = getattr(
                        client.count(collection_name=name, exact=False),
                        "count",
                        None,
                    )
                except Exception as exc:
                    partial_errors.append(self._error_message(exc))
                rows.append(
                    {
                        "name": name,
                        "point_count": point_count,
                        "role": self._qdrant_collection_role(name),
                    }
                )
            payload["collections"] = rows
            if partial_errors:
                payload["status"] = "partial"
                payload["message"] = (
                    f"Collections loaded; count failed: {'; '.join(partial_errors)}"
                )
        except Exception as exc:
            payload["status"] = "error"
            payload["message"] = self._error_message(exc)
        return payload

    def _ai_status(self) -> dict[str, Any]:
        key_valid = not bool(self.settings.is_openai_api_key_placeholder)
        payload: dict[str, Any] = {
            "status": "ok" if key_valid else "error",
            "provider": "openai",
            "llm_model": self.settings.LLM_MODEL,
            "embedding_model": self.settings.EMBEDDING_MODEL,
            "embedding_dimensions": self.settings.effective_vector_dimensions,
            "api_mode": (
                "responses"
                if self.settings.OPENAI_USE_RESPONSES
                else "chat_completions"
            ),
            "key_valid": key_valid,
            "initialization": {
                "status": "skipped",
                "message": "Skipped because OPENAI_API_KEY is missing or placeholder.",
            },
            "models_api": {
                "status": "skipped",
                "message": "Skipped because OPENAI_API_KEY is missing or placeholder.",
                "model_count": 0,
                "embedding_model_available": False,
            },
        }
        if not key_valid:
            return payload

        payload["initialization"] = self._ai_initialization_status()
        payload["models_api"] = self._models_registry_status()
        if (
            payload["initialization"]["status"] != "ok"
            or payload["models_api"]["status"] != "ok"
        ):
            payload["status"] = "error"
        return payload

    def _ai_initialization_status(self) -> dict[str, str]:
        try:
            self.ai_initializer()
            return {"status": "ok", "message": "Provider initialized."}
        except Exception as exc:
            return {"status": "error", "message": self._error_message(exc)}

    def list_available_models(self) -> dict[str, Any]:
        """Return LLM and embedding models from the registry.

        ``configured_providers`` lists embedding providers whose API key is set.
        ``configured_llm_providers`` lists LLM providers whose API key is set.
        The UI uses these to filter dropdown options to only runnable models.
        """
        configured_default = self.settings.LLM_MODEL
        embedding_models = [
            {
                "id": e.id,
                "provider": e.provider,
                "dimensions": e.dimensions,
                "display_name": e.display_name,
                "default": e.id == self.settings.EMBEDDING_MODEL,
            }
            for e in embedding_manager.list_models()
        ]
        llm_models = [
            {
                "id": e.id,
                "provider": e.provider,
                "display_name": e.display_name,
                "context_window": e.context_window,
                "supports_reasoning": e.supports_reasoning,
                "supports_vision": e.supports_vision,
                "default": e.id == self.settings.LLM_MODEL,
            }
            for e in llm_manager.list_models()
        ]
        has_openai = self._has_secret(self.settings.openai_api_key)
        has_gemini = self._has_secret(self.settings.gemini_api_key)
        configured_providers: list[str] = []
        if has_openai:
            configured_providers.append("openai")
        if has_gemini:
            configured_providers.append("gemini")
        configured_llm_providers: list[str] = []
        if has_openai:
            configured_llm_providers.append("openai")
        if self._has_secret(self.settings.anthropic_api_key):
            configured_llm_providers.append("anthropic")
        if has_gemini:
            configured_llm_providers.append("gemini")
        return {
            "models": [{"id": configured_default, "default": True}],
            "configured_default": configured_default,
            "embedding_models": embedding_models,
            "configured_providers": configured_providers,
            "llm_models": llm_models,
            "configured_llm_providers": configured_llm_providers,
        }

    def _models_registry_status(self) -> dict[str, Any]:
        """Check whether configured models appear in the registry."""
        return {
            "status": "ok",
            "message": "Registry check succeeded.",
            "model_count": len(EMBEDDING_REGISTRY_IDS),
            "embedding_model_available": self.settings.EMBEDDING_MODEL in EMBEDDING_REGISTRY_IDS,
            "llm_model_count": len(LLM_REGISTRY_IDS),
        }

    def _database_target(self, mode: str) -> dict[str, str | None]:
        if mode == "snowflake":
            return {
                "account": self.settings.SNOWFLAKE_ACCOUNT,
                "user": self.settings.SNOWFLAKE_USER,
                "database": self.settings.SNOWFLAKE_DATABASE,
                "schema": self.settings.SNOWFLAKE_SCHEMA,
                "warehouse": self.settings.SNOWFLAKE_WAREHOUSE,
                "role": self.settings.SNOWFLAKE_ROLE,
            }

        url = getattr(self.engine, "url", None)
        database = getattr(url, "database", None) if url is not None else None
        return {"path": database or "./rag_local.db"}

    def _qdrant_collection_role(self, name: str) -> str:
        if name == self.settings.COLLECTION_NAME:
            return "documents"
        if name == self.settings.CHAT_HISTORY_COLLECTION_NAME:
            return "chat_history"
        return "other"

    @staticmethod
    def _has_secret(value: str | None) -> bool:
        key = (value or "").strip().strip("'\"").strip()
        return bool(key)

    @staticmethod
    def _overall_status(section: dict[str, Any]) -> str:
        return str(section.get("status") or "unknown")

    @staticmethod
    def _error_message(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"
