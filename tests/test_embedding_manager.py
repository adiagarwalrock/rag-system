"""Tests for the EmbeddingProvider hierarchy and EmbeddingManager."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.core.models.embedding.base import (
    EMBEDDING_REGISTRY,
    EmbeddingModelEntry,
    EmbeddingProvider,
)


# ---------------------------------------------------------------------------
# EmbeddingProvider.detect_provider
# ---------------------------------------------------------------------------


def test_detect_provider_registry_lookup_openai():
    assert EmbeddingProvider.detect_provider("text-embedding-3-large") == "openai"


def test_detect_provider_registry_lookup_gemini():
    assert EmbeddingProvider.detect_provider("gemini-embedding-2") == "gemini"


def test_detect_provider_gemini_prefix_fallback():
    # Model not in registry but has gemini- prefix
    assert EmbeddingProvider.detect_provider("gemini-new-model-xyz") == "gemini"


def test_detect_provider_models_gemini_prefix_fallback():
    assert EmbeddingProvider.detect_provider("models/gemini-embedding-v99") == "gemini"


def test_detect_provider_defaults_to_openai():
    assert EmbeddingProvider.detect_provider("some-unknown-model") == "openai"


def test_detect_provider_unknown_model_defaults_to_openai():
    assert EmbeddingProvider.detect_provider("completely-unknown") == "openai"


# ---------------------------------------------------------------------------
# EmbeddingProvider.get_entry
# ---------------------------------------------------------------------------


def test_get_entry_known_model():
    entry = EmbeddingProvider.get_entry("text-embedding-3-large")
    assert entry is not None
    assert entry.id == "text-embedding-3-large"
    assert entry.provider == "openai"
    assert entry.dimensions == 3072


def test_get_entry_unknown_model():
    assert EmbeddingProvider.get_entry("does-not-exist") is None


# ---------------------------------------------------------------------------
# EMBEDDING_REGISTRY contents
# ---------------------------------------------------------------------------


def test_registry_has_openai_and_gemini_entries():
    providers = {e.provider for e in EMBEDDING_REGISTRY}
    assert "openai" in providers
    assert "gemini" in providers


def test_registry_all_entries_have_positive_dimensions():
    for entry in EMBEDDING_REGISTRY:
        assert entry.dimensions > 0, f"{entry.id} has non-positive dimensions"


def test_registry_no_duplicate_ids():
    ids = [e.id for e in EMBEDDING_REGISTRY]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# OpenAIEmbeddingProvider.build
# ---------------------------------------------------------------------------


def test_openai_provider_build_without_dimensions():
    from app.core.models.embedding.openai import OpenAIEmbeddingProvider

    fake_embed = MagicMock()
    with patch(
        "app.core.models.embedding.openai.OpenAIEmbedding", return_value=fake_embed
    ) as mock_cls:
        provider = OpenAIEmbeddingProvider(model_id="text-embedding-3-large", api_key="sk-test")
        result = provider.build()

    mock_cls.assert_called_once_with(model="text-embedding-3-large", api_key="sk-test")
    assert result is fake_embed


def test_openai_provider_build_with_dimensions():
    from app.core.models.embedding.openai import OpenAIEmbeddingProvider

    fake_embed = MagicMock()
    with patch(
        "app.core.models.embedding.openai.OpenAIEmbedding", return_value=fake_embed
    ) as mock_cls:
        provider = OpenAIEmbeddingProvider(
            model_id="text-embedding-3-large", api_key="sk-test", dimensions=1536
        )
        result = provider.build()

    mock_cls.assert_called_once_with(
        model="text-embedding-3-large", api_key="sk-test", dimensions=1536
    )
    assert result is fake_embed


# ---------------------------------------------------------------------------
# GeminiEmbeddingProvider.build
# ---------------------------------------------------------------------------


def test_gemini_provider_build():
    from app.core.models.embedding.gemini import GeminiEmbeddingProvider

    fake_embed = MagicMock()
    with patch(
        "app.core.models.embedding.gemini.GoogleGenAIEmbedding", return_value=fake_embed
    ) as mock_cls:
        provider = GeminiEmbeddingProvider(model_id="text-embedding-004", api_key="gm-test")
        result = provider.build()

    mock_cls.assert_called_once_with(model_name="text-embedding-004", api_key="gm-test")
    assert result is fake_embed


# ---------------------------------------------------------------------------
# EmbeddingManager
# ---------------------------------------------------------------------------


@pytest.fixture()
def manager():
    from app.core.embedding_manager import EmbeddingManager

    return EmbeddingManager()


def test_manager_caches_same_instance_for_same_key(manager):
    fake = MagicMock()
    with patch.object(manager, "_build", return_value=fake) as mock_build:
        inst1 = manager.get_instance(model_id="text-embedding-3-large", api_key="k")
        inst2 = manager.get_instance(model_id="text-embedding-3-large", api_key="k")

    mock_build.assert_called_once()
    assert inst1 is inst2


def test_manager_builds_different_instances_for_different_models(manager):
    fake_a, fake_b = MagicMock(), MagicMock()
    build_returns = [fake_a, fake_b]

    with patch.object(manager, "_build", side_effect=build_returns):
        inst_a = manager.get_instance(model_id="text-embedding-3-large", api_key="k")
        inst_b = manager.get_instance(model_id="text-embedding-3-small", api_key="k")

    assert inst_a is fake_a
    assert inst_b is fake_b


def test_manager_builds_different_instances_for_different_keys(manager):
    fake_a, fake_b = MagicMock(), MagicMock()

    with patch.object(manager, "_build", side_effect=[fake_a, fake_b]):
        inst_a = manager.get_instance(model_id="text-embedding-3-large", api_key="k1")
        inst_b = manager.get_instance(model_id="text-embedding-3-large", api_key="k2")

    assert inst_a is fake_a
    assert inst_b is fake_b


def test_manager_invalidate_single_model(manager):
    fake_a, fake_b, fake_a2 = MagicMock(), MagicMock(), MagicMock()

    with patch.object(manager, "_build", side_effect=[fake_a, fake_b, fake_a2]):
        manager.get_instance(model_id="model-a", api_key="k")
        manager.get_instance(model_id="model-b", api_key="k")

    manager.invalidate("model-a")

    with patch.object(manager, "_build", return_value=fake_a2):
        new_a = manager.get_instance(model_id="model-a", api_key="k")

    assert new_a is fake_a2
    # model-b still cached
    with patch.object(manager, "_build") as no_build:
        manager.get_instance(model_id="model-b", api_key="k")
        no_build.assert_not_called()


def test_manager_invalidate_all(manager):
    fake = MagicMock()
    with patch.object(manager, "_build", return_value=fake):
        manager.get_instance(model_id="model-a", api_key="k")
        manager.get_instance(model_id="model-b", api_key="k")

    manager.invalidate()
    assert manager._cache == {}


def test_manager_list_models_returns_registry(manager):
    models = manager.list_models()
    assert isinstance(models, list)
    ids = [m.id for m in models]
    assert "text-embedding-3-large" in ids
    assert "gemini-embedding-2" in ids


def test_manager_raises_for_unknown_provider(manager):
    with patch(
        "app.core.embedding_manager.EmbeddingProvider.detect_provider",
        return_value="unknown_provider",
    ):
        with pytest.raises(ValueError, match="No EmbeddingProvider registered"):
            manager.get_instance(model_id="weird-model", api_key="k")
