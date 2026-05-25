"""
Cross-encoder semantic reranking for retrieval candidates.
"""

import logging
from collections.abc import Sequence
from threading import Lock
from typing import Any

from llama_index.core.schema import MetadataMode

from app.core.safe_coerce import safe_float

logger = logging.getLogger(__name__)


class CrossEncoderSemanticReranker:
    """Lazily loads and reuses a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        *,
        model_name: str,
        fallback_model_name: str,
        hf_api_token: str = "",
        device: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name if hf_api_token else fallback_model_name
        self.hf_api_token = hf_api_token
        self.device = device
        self.trust_remote_code = trust_remote_code
        self._model: Any | None = None
        self._load_error: str | None = None
        self._lock = Lock()

    def rerank(self, query: str, nodes: Sequence, top_k: int) -> list:
        if not query or not nodes:
            return list(nodes[:top_k])

        model = self._get_model()
        if model is None:
            return list(nodes[:top_k])

        pairs = [(query, _node_content(node)) for node in nodes]
        try:
            cross_encoder_scores = model.predict(pairs)
        except Exception:
            logger.exception("Cross-encoder reranking failed; using metadata rank")
            return list(nodes[:top_k])

        if len(cross_encoder_scores) != len(nodes):
            logger.warning(
                "Cross-encoder returned %d scores for %d nodes; using metadata rank",
                len(cross_encoder_scores),
                len(nodes),
            )
            return list(nodes[:top_k])

        metadata_scores = [safe_float(node.score, 0.0) or 0.0 for node in nodes]
        semantic_scores = [
            safe_float(cross_encoder_score, metadata_scores[index]) or 0.0
            for index, cross_encoder_score in enumerate(cross_encoder_scores)
        ]
        normalized_metadata_scores = _minmax(metadata_scores)
        normalized_semantic_scores = _minmax(semantic_scores)

        for index, (node, cross_encoder_score) in enumerate(
            zip(nodes, cross_encoder_scores)
        ):
            metadata = node.node.metadata or {}
            metadata_score = metadata_scores[index]
            semantic_score = semantic_scores[index]
            metadata["retrieval_score"] = metadata_score
            metadata["cross_encoder_score"] = semantic_score
            node.node.metadata = metadata
            node.score = (
                normalized_semantic_scores[index] * 0.4
                + normalized_metadata_scores[index] * 0.6
            )

        ranked = sorted(nodes, key=lambda node: node.score or 0.0, reverse=True)
        return list(ranked[:top_k])

    def _get_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None

        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                return None

            try:
                from sentence_transformers import CrossEncoder

                kwargs: dict[str, Any] = {
                    "trust_remote_code": self.trust_remote_code,
                }
                if self.hf_api_token:
                    kwargs["token"] = self.hf_api_token
                if self.device:
                    kwargs["device"] = self.device
                self._model = CrossEncoder(self.model_name, **kwargs)
            except Exception as exc:
                self._load_error = str(exc)
                logger.warning(
                    "Cross-encoder model %s is unavailable; using metadata rank. "
                    "Install sentence-transformers and ensure the model is cached "
                    "or reachable. Error: %s",
                    self.model_name,
                    exc,
                )
                return None

        return self._model


def _node_content(node) -> str:
    try:
        return node.node.get_content(metadata_mode=MetadataMode.EMBED)
    except Exception:
        return node.node.text or ""


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [1.0 for _ in values]
    span = max_value - min_value
    return [(value - min_value) / span for value in values]
