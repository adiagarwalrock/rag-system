"""
Cross-encoder semantic reranking for retrieval candidates.
"""

import logging
from collections.abc import Sequence
from threading import Lock
from typing import Any

from llama_index.core.schema import MetadataMode

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

        for node, cross_encoder_score in zip(nodes, cross_encoder_scores):
            metadata = node.node.metadata or {}
            metadata["retrieval_score"] = node.score
            node.node.metadata = metadata
            node.score = _safe_score(cross_encoder_score, node.score)

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


def _safe_score(value, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
