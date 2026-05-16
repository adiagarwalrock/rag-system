from types import SimpleNamespace
from typing import cast

from app.db.models.chat import ChatMessage
from app.core.config import settings
from observability.cost_tracker import count_tokens
from app.ingestion.pdf_pipeline.chunk_builder import split_body_text
from app.ingestion.pdf_pipeline.registry import PDFPipelineRegistry
from app.services.chat_conversation_service import _heuristic_summary


def test_split_body_text_uses_token_limits(monkeypatch):
    monkeypatch.setattr(settings, "BODY_TEXT_CHUNK_MAX_TOKENS", 24)
    monkeypatch.setattr(settings, "BODY_TEXT_CHUNK_OVERLAP_TOKENS", 6)
    monkeypatch.setattr(settings, "TOKEN_BUDGET_ENCODING", "o200k_base")

    # Force deterministic token-window fallback to test token budgeting directly.
    monkeypatch.setattr(
        PDFPipelineRegistry().chunker_manager,
        "create_sentence_chunker",
        lambda **_kwargs: None,
    )

    text = " ".join(["Revenue grew 12 percent year over year."] * 40)
    chunks = split_body_text(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert (
            count_tokens(chunk, encoding_name=settings.TOKEN_BUDGET_ENCODING)
            <= settings.BODY_TEXT_CHUNK_MAX_TOKENS
        )


def test_heuristic_summary_respects_token_limit(monkeypatch):
    monkeypatch.setattr(settings, "CHAT_SUMMARY_MAX_OUTPUT_TOKENS", 20)
    monkeypatch.setattr(settings, "TOKEN_BUDGET_ENCODING", "o200k_base")

    rows = [
        SimpleNamespace(role="user", content=" ".join(["x"] * 300)),
        SimpleNamespace(role="assistant", content=" ".join(["y"] * 300)),
    ]

    summary = _heuristic_summary(cast(list[ChatMessage], rows))
    assert (
        count_tokens(summary, encoding_name=settings.TOKEN_BUDGET_ENCODING)
        <= settings.CHAT_SUMMARY_MAX_OUTPUT_TOKENS
    )
