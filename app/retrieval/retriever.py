"""
Retriever: orchestrates the full retrieval pipeline.

Flow: vector retrieval -> ranking -> evidence selection -> conflict checks -> citation building
"""

import logging
import mimetypes
import re
import hashlib
import base64
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import httpx as _httpx
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from app.core.ai_provider import (
    extract_chat_response_text,
    extract_chat_response_reasoning,
    get_llm,
    invoke_llm_chat,
    stream_invoke_llm_chat,
    normalize_reasoning_effort,
    normalize_reasoning_summary,
)
from app.core.config import settings
from app.core.safe_coerce import normalize_metric_subject, safe_bool, safe_int
from app.core.prompts import (
    GROUNDED_ANSWER_DEVELOPER_PROMPT,
    build_grounded_answer_prompt,
)
from app.core.token_budget import ResponsesInputBudgeter
from app.indexing.vector_store import vector_store_manager
from app.retrieval.citation_builder import (
    build_citations,
    format_enriched_metadata_for_prompt,
)
from app.retrieval.conflict_detector import detect_conflicts
from app.retrieval.query_intent import (
    RetrievalIntent,
    analyze_retrieval_intent,
    query_entity_aliases,
)
from app.retrieval.query_expansion import build_query_variants, should_expand_query
from app.retrieval.reranker import rerank_nodes

logger = logging.getLogger(__name__)

DEFAULT_EVIDENCE_LIMIT = 15
COMPARATIVE_EVIDENCE_LIMIT = 20
CONFLICT_EVIDENCE_LIMIT = 20
MAX_MULTIMODAL_IMAGES = 6
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
REASONING_CHUNK_TYPES = {
    "reasoning_table",
    "reasoning_chart",
    "reasoning_figure",
    "reasoning_page",
}
REASONING_PRIORITY_TERMS = (
    "chart",
    "charts",
    "graph",
    "graphs",
    "table",
    "tables",
    "figure",
    "figures",
    "diagram",
    "diagrams",
    "image",
    "images",
    "screenshot",
    "screenshots",
    "visual",
    "trend",
    "trends",
    "why",
    "reason",
    "analysis",
    "explain",
)
COMPARATIVE_TERMS = (
    "compare",
    "comparison",
    "difference",
    "change",
    "changes",
    "changed",
    "conflict",
    "conflicting",
    "contradict",
    "versus",
    "vs",
    "between",
)
BROAD_BALANCED_TERMS = (
    "different",
    "both",
    "each",
    "across",
    "sectors",
    "represented",
    "by reit",
    "for each",
    "in the corpus",
    "corpus",
)
VISUAL_QUERY_TERMS = (
    "image",
    "images",
    "figure",
    "figures",
    "chart",
    "charts",
    "graph",
    "graphs",
    "diagram",
    "diagrams",
    "map",
    "maps",
    "screenshot",
    "screenshots",
    "visual",
)
TABLE_EVIDENCE_TERMS = (
    "table",
    "matrix",
    "tabular",
    "top ",
    "top-",
    "breakdown",
    "market mix",
    "portfolio composition",
)
CHART_EVIDENCE_TERMS = (
    "chart",
    "graph",
    "plot",
    "legend",
    "infographic",
    "pie",
    "trend",
    "line",
    "bar",
)
TIME_ANCHORED_TERMS = (
    "as of",
    "q1",
    "q2",
    "q3",
    "q4",
    "fy",
    "fiscal",
    "expected close",
)
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
RECENT_TURN_MAX_CHARS = 320
CROSS_SESSION_USER_MAX_CHARS = 220
CROSS_SESSION_ASSISTANT_MAX_CHARS = 260


@dataclass(frozen=True, slots=True)
class GroundedAnswerResult:
    answer: str
    reasoning: str | None
    images_used: list[str]
    reasoning_effort_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "reasoning": self.reasoning,
            "images_used": self.images_used,
            "reasoning_effort_applied": self.reasoning_effort_applied,
        }


class GroundedAnswerSynthesizer:
    """Synthesize grounded answers from selected evidence and conflicts."""

    def __init__(
        self,
        *,
        client_id: str,
        reasoning_effort: str,
        reasoning_summary: str | None = None,
        reasoning_callback: Callable[[str], None] | None = None,
        conversation_context: dict[str, Any],
    ):
        self.client_id = client_id
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.reasoning_callback = reasoning_callback
        self.conversation_context = conversation_context

    def synthesize(
        self,
        *,
        question: str,
        citations: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        effort_applied = settings.OPENAI_USE_RESPONSES

        if not citations:
            return GroundedAnswerResult(
                answer=(
                    "I could not find enough relevant evidence in the uploaded "
                    "documents to answer this question confidently."
                ),
                reasoning=None,
                images_used=[],
                reasoning_effort_applied=effort_applied,
            ).to_dict()

        image_paths = _collect_image_evidence_paths(citations)

        if settings.OPENAI_USE_RESPONSES:
            responses_result = self._try_responses_synthesis(
                question=question,
                citations=citations,
                conflicts=conflicts,
                image_paths=image_paths,
                effort_applied=effort_applied,
            )
            if responses_result is not None:
                return responses_result.to_dict()

        chat_result = self._try_chat_synthesis(
            question=question,
            citations=citations,
            conflicts=conflicts,
            image_paths=image_paths,
            effort_applied=effort_applied,
        )
        if chat_result is not None:
            return chat_result.to_dict()

        logger.error("Answer synthesis failed; returning source-grounded fallback")
        return GroundedAnswerResult(
            answer=_build_source_grounded_fallback(citations),
            reasoning=None,
            images_used=image_paths,
            reasoning_effort_applied=effort_applied,
        ).to_dict()

    def _try_responses_synthesis(
        self,
        *,
        question: str,
        citations: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
        image_paths: list[str],
        effort_applied: bool,
    ) -> GroundedAnswerResult | None:
        try:
            budgeter = ResponsesInputBudgeter(model=settings.LLM_MODEL)
            sections = _build_labeled_context_sections(
                question=question,
                citations=citations,
                conflicts=conflicts,
                conversation_context=self.conversation_context,
            )
            input_messages, _, metrics = budgeter.build_budgeted_sections(
                developer_prompt=GROUNDED_ANSWER_DEVELOPER_PROMPT,
                question=question,
                recent_turns=self.conversation_context.get("recent_turns") or [],
                session_summary=sections["session_summary"],
                cross_session_lines=sections["cross_session_lines"],
                evidence_lines=sections["evidence_lines"],
                conflict_lines=sections["conflict_lines"],
            )
            logger.info(
                "Responses budget usage model=%s input_tokens=%d/%d history=%d summary=%d cross=%d evidence=%d conflict=%d",
                metrics.model,
                metrics.total_input_tokens,
                metrics.input_budget_tokens,
                metrics.history_tokens,
                metrics.summary_tokens,
                metrics.cross_session_tokens,
                metrics.evidence_tokens,
                metrics.conflict_tokens,
            )
            input_messages, used_image_paths = _append_image_inputs(
                input_messages=input_messages,
                image_paths=image_paths,
            )

            llm_kwargs = dict(
                model=settings.LLM_MODEL,
                input_messages=input_messages,
                reasoning_effort=self.reasoning_effort,
                reasoning_summary=self.reasoning_summary,
                max_output_tokens=settings.RESPONSE_MAX_OUTPUT_TOKENS,
                prompt_cache_key=settings.RESPONSE_PROMPT_CACHE_KEY,
                prompt_cache_retention=settings.RESPONSE_PROMPT_CACHE_RETENTION,
                safety_identifier=f"{settings.RESPONSE_SAFETY_IDENTIFIER_PREFIX}:{self.client_id}",
                user_tag=settings.RESPONSE_USER_TAG,
                timeout_seconds=settings.RESPONSE_SYNTHESIS_TIMEOUT_SECONDS,
            )

            logger.info(
                "Synthesis path: streaming=%s reasoning_effort=%s reasoning_summary=%s use_responses=%s",
                self.reasoning_callback is not None,
                self.reasoning_effort,
                self.reasoning_summary,
                settings.OPENAI_USE_RESPONSES,
            )

            if self.reasoning_callback is not None:
                answer_parts: list[str] = []
                reasoning_parts: list[str] = []
                delta_count = 0
                reasoning_delta_count = 0
                for r_delta, a_delta in stream_invoke_llm_chat(**llm_kwargs):
                    delta_count += 1
                    if r_delta:
                        reasoning_delta_count += 1
                        reasoning_parts.append(r_delta)
                        try:
                            self.reasoning_callback(r_delta)
                        except Exception:
                            pass
                    if a_delta:
                        answer_parts.append(a_delta)
                text = "".join(answer_parts).strip()
                reasoning_text: str | None = "".join(reasoning_parts).strip() or None
                logger.info(
                    "Streaming synthesis complete: total_deltas=%d reasoning_deltas=%d answer_chars=%d reasoning_chars=%d",
                    delta_count,
                    reasoning_delta_count,
                    len(text),
                    len(reasoning_text or ""),
                )
                answer, parsed_reasoning = _split_reasoning_from_text(text)
                reasoning: str | None = reasoning_text or parsed_reasoning
            else:
                response = invoke_llm_chat(**llm_kwargs)
                text = extract_chat_response_text(response)
                answer, tag_reasoning = _split_reasoning_from_text(text)
                # Prefer the ThinkingBlock reasoning summary (from OpenAI's
                # reasoning.summary feature) over any <thinking> tag fallback.
                block_reasoning = extract_chat_response_reasoning(response)
                logger.info(
                    "Blocking synthesis complete: answer_chars=%d block_reasoning_chars=%d tag_reasoning_chars=%d",
                    len(text),
                    len(block_reasoning or ""),
                    len(tag_reasoning or ""),
                )
                reasoning: str | None = block_reasoning or tag_reasoning

            if not answer:
                raise ValueError("LLM returned empty answer")
            return GroundedAnswerResult(
                answer=answer,
                reasoning=reasoning,
                images_used=used_image_paths,
                reasoning_effort_applied=effort_applied,
            )
        except Exception as exc:
            # Re-raise timeout errors — don't fall through to the chat path and
            # wait another 240s when the API is just slow.
            if isinstance(exc, (TimeoutError, _httpx.TimeoutException)):
                raise
            logger.exception(
                "Responses answer synthesis failed; falling back to LlamaIndex chat"
            )
            return None

    def _try_chat_synthesis(
        self,
        *,
        question: str,
        citations: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
        image_paths: list[str],
        effort_applied: bool,
    ) -> GroundedAnswerResult | None:
        llm = get_llm(
            reasoning_effort=self.reasoning_effort,
            timeout_seconds=settings.RESPONSE_SYNTHESIS_TIMEOUT_SECONDS,
        )
        prompt = _build_grounded_prompt(
            question,
            citations,
            conflicts,
            image_attachment_count=len(image_paths),
            conversation_context=self.conversation_context,
        )

        for attempt_name, message, used_images in self._chat_attempts(
            prompt, image_paths
        ):
            try:
                response = llm.chat([message])
                answer, reasoning = _extract_answer_and_reasoning_from_chat(response)
                if not answer:
                    raise ValueError("LLM returned empty answer")
                return GroundedAnswerResult(
                    answer=answer,
                    reasoning=reasoning,
                    images_used=used_images,
                    reasoning_effort_applied=effort_applied,
                )
            except Exception:
                logger.exception("%s answer synthesis failed", attempt_name)

        return None

    @staticmethod
    def _chat_attempts(
        prompt: str,
        image_paths: list[str],
    ) -> list[tuple[str, ChatMessage, list[str]]]:
        attempts: list[tuple[str, ChatMessage, list[str]]] = []
        if image_paths:
            attempts.append(
                (
                    "multimodal",
                    _build_grounded_message(prompt, image_paths),
                    image_paths,
                )
            )
        attempts.append(("text_only", _build_grounded_message(prompt, []), []))
        return attempts


class VecteraRetriever:
    """
    Full retrieval pipeline with client-scoped filtering,
    reranking, temporal awareness, conflict detection, and citations.
    """

    def __init__(
        self,
        client_id: str,
        top_k: int = 15,
        reasoning_effort: str = "medium",
        reasoning_summary: str | None = None,
        conversation_context: dict[str, Any] | None = None,
        reasoning_callback: Callable[[str], None] | None = None,
    ):
        self.client_id = client_id
        self.top_k = top_k
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self.conversation_context = conversation_context or {}
        self.prefetch_top_k = top_k * 5
        self.evidence_limit = DEFAULT_EVIDENCE_LIMIT
        self.comparative_evidence_limit = COMPARATIVE_EVIDENCE_LIMIT
        self.conflict_evidence_limit = CONFLICT_EVIDENCE_LIMIT
        self._last_retrieval_metadata: dict[str, Any] = {}
        self.filters = MetadataFilters(
            filters=[ExactMatchFilter(key="client_id", value=self.client_id)]
        )
        self.answer_synthesizer = GroundedAnswerSynthesizer(
            client_id=self.client_id,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=normalize_reasoning_summary(reasoning_summary),
            reasoning_callback=reasoning_callback,
            conversation_context=self.conversation_context,
        )

    def query(
        self,
        question: str,
        status_callback: Callable[[str], None] | None = None,
    ) -> Dict[str, Any]:
        """
        Execute the full retrieval and answer pipeline.

        ``status_callback`` is an optional callable invoked with a short human-readable
        label at each pipeline stage.  The UI uses this to update a live progress panel
        while the synchronous pipeline is running in a background thread.

        Returns:
            Dict with answer, citations, conflicts, and metadata.
        """
        def _emit(msg: str) -> None:
            if status_callback is not None:
                try:
                    status_callback(msg)
                except Exception:
                    pass  # never let a UI callback crash the pipeline

        logger.info("Query for client %s: %s", self.client_id, question[:100])

        _emit("🔍 Retrieving relevant chunks…")
        source_nodes, retrieval_metadata = self._retrieve(question)
        intent = analyze_retrieval_intent(question)

        if not source_nodes:
            _emit("⚠️ No matching chunks found in the document index")
            return {
                "answer": "I could not find relevant information in the uploaded documents to answer this question.",
                "reasoning": None,
                "citations": [],
                "conflicts": [],
                "source_count": 0,
                "images_used": [],
                "image_evidence_count": 0,
                "reasoning_effort": self.reasoning_effort,
                "reasoning_effort_applied": False,
                "retrieval_diagnostics": _build_retrieval_diagnostics([], []),
                "intent_labels": list(intent.labels),
                "companion_queries": [],
                "companion_counts_by_query": {},
                "evidence_by_document": {},
                "evidence_by_version_group": {},
                "evidence_by_entity": {},
                **retrieval_metadata,
            }

        _emit(f"📊 Reranking {len(source_nodes)} candidate chunks…")
        ranked_nodes = self._rank_nodes(question, source_nodes)

        _emit("✂️ Selecting evidence…")
        evidence_nodes = self._select_evidence_nodes(question, ranked_nodes)
        retrieval_diagnostics = _build_retrieval_diagnostics(
            ranked_nodes, evidence_nodes, intent=intent
        )

        conflicts = detect_conflicts(
            ranked_nodes, evidence_nodes=evidence_nodes, question=question
        )

        citations = build_citations(evidence_nodes)

        _emit("🧠 Synthesizing grounded answer…")
        synthesis = self._synthesize_answer(question, citations, conflicts)

        return {
            "answer": synthesis["answer"],
            "reasoning": synthesis.get("reasoning"),
            "citations": citations,
            "conflicts": conflicts,
            "source_count": len(ranked_nodes),
            "evidence_count": len(evidence_nodes),
            "images_used": synthesis.get("images_used", []),
            "image_evidence_count": len(synthesis.get("images_used", [])),
            "reasoning_effort": self.reasoning_effort,
            "reasoning_effort_applied": synthesis.get(
                "reasoning_effort_applied", False
            ),
            "retrieval_diagnostics": retrieval_diagnostics,
            "intent_labels": list(intent.labels),
            "evidence_by_document": _node_counts_by_document(evidence_nodes),
            "evidence_by_version_group": _node_counts_by_version_group(evidence_nodes),
            "evidence_by_entity": _node_counts_by_entity(evidence_nodes, intent),
            **retrieval_metadata,
        }

    def retrieve_only(self, question: str) -> List:
        """Retrieve source nodes without generating an answer."""
        source_nodes, _ = self._retrieve(question)
        return self._rank_nodes(question, source_nodes)

    def _rank_nodes(self, question: str, source_nodes: list) -> list:
        intent = analyze_retrieval_intent(question)
        comparative_query = _is_comparison_or_conflict_query(question)
        balanced_query = _needs_balanced_evidence_query(question)
        conflict_focused_query = _is_conflict_focused_query(question)
        prefer_latest = not (
            comparative_query
            or balanced_query
            or intent.has("temporal_delta")
            or _is_time_anchored_query(question)
        )
        rank_top_k = self._resolve_rank_top_k(
            comparative_query=comparative_query
            or balanced_query
            or intent.has("temporal_delta")
            or intent.has("outlook_scope"),
            conflict_focused_query=conflict_focused_query,
            high_diversity_query=_is_high_diversity_intent(intent),
        )
        return rerank_nodes(
            source_nodes,
            top_k=rank_top_k,
            prefer_latest=prefer_latest,
            query=question,
        )

    def _retrieve(self, question: str) -> tuple[list, dict[str, Any]]:
        try:
            nodes, expanded = self._retrieve_with_mode(question, hybrid=True)
            mode = "hybrid"
        except Exception:
            logger.exception(
                "Hybrid retrieval failed for client %s; falling back to dense retrieval",
                self.client_id,
            )
            nodes, expanded = self._retrieve_with_mode(question, hybrid=False)
            mode = "dense_fallback"

        logger.info(
            "Retrieved %d nodes with mode=%s expansion=%s for client %s",
            len(nodes),
            mode,
            expanded,
            self.client_id,
        )
        return nodes, {
            "retrieval_mode": mode,
            "query_expanded": expanded,
            **self._last_retrieval_metadata,
        }

    def _retrieve_with_mode(self, question: str, hybrid: bool) -> tuple[list, bool]:
        intent = analyze_retrieval_intent(question)
        self._last_retrieval_metadata = _empty_retrieval_intent_metadata(intent)
        prefetch_top_k = (
            self.prefetch_top_k + 10
            if _is_conflict_focused_query(question) or intent.companion_queries
            else self.prefetch_top_k
        )
        base_retriever = vector_store_manager.get_retriever(
            filters=self.filters,
            similarity_top_k=prefetch_top_k,
            sparse_top_k=prefetch_top_k,
            hybrid_top_k=prefetch_top_k,
            hybrid=hybrid,
        )

        expansion_question = self._build_query_expansion_input(question)
        should_expand = should_expand_query(expansion_question)
        companion_queries = list(intent.companion_queries)
        if should_expand:
            nodes, expanded, companion_counts = self._retrieve_with_expansion(
                question,
                expansion_question,
                base_retriever,
                companion_queries=companion_queries,
            )
            self._last_retrieval_metadata = _retrieval_intent_metadata(
                intent=intent,
                companion_queries=companion_queries,
                companion_counts=companion_counts,
            )
            return nodes, expanded

        if companion_queries:
            queries = [question, *companion_queries]
            batches = _retrieve_query_batches(base_retriever, queries)
            self._last_retrieval_metadata = _retrieval_intent_metadata(
                intent=intent,
                companion_queries=companion_queries,
                companion_counts={
                    query: len(batch)
                    for query, batch in zip(queries, batches)
                    if query in companion_queries
                },
            )
            return _fuse_node_batches(batches), False

        return base_retriever.retrieve(question), False

    def _retrieve_with_expansion(
        self,
        question: str,
        expansion_question: dict[str, Any],
        base_retriever,
        *,
        companion_queries: list[str] | None = None,
    ) -> tuple[list, bool, dict[str, int]]:
        query_variants = build_query_variants(expansion_question)
        companion_queries = companion_queries or []
        queries = _dedupe_retrieval_queries([*query_variants, *companion_queries])
        if len(queries) == 1:
            return base_retriever.retrieve(question), False, {}

        batches = _retrieve_query_batches(base_retriever, queries)
        companion_counts = {
            query: len(batch)
            for query, batch in zip(queries, batches)
            if query in companion_queries
        }
        return _fuse_node_batches(batches), len(query_variants) > 1, companion_counts

    def _build_query_expansion_input(self, question: str) -> dict[str, Any]:
        return {
            "current_question": question,
            "recent_turns": self.conversation_context.get("recent_turns") or [],
        }

    def _select_evidence_nodes(self, question: str, ranked_nodes: list) -> list:
        if not ranked_nodes:
            return []

        intent = analyze_retrieval_intent(question)
        comparative_query = _is_comparison_or_conflict_query(question)
        balanced_query = _needs_balanced_evidence_query(question)
        conflict_focused_query = _is_conflict_focused_query(question)
        evidence_cap = self._resolve_evidence_cap(
            comparative_query=comparative_query
            or balanced_query
            or intent.has("temporal_delta")
            or intent.has("outlook_scope"),
            conflict_focused_query=conflict_focused_query,
        )
        candidate_nodes = self._prioritize_conflict_candidates(
            ranked_nodes=ranked_nodes,
            conflict_focused_query=conflict_focused_query,
        )
        primary_candidates, secondary_candidates = self._partition_reasoning_candidates(
            question=question,
            candidate_nodes=candidate_nodes,
        )
        selected = self._collect_evidence_candidates(
            primary_candidates=primary_candidates,
            secondary_candidates=secondary_candidates,
            evidence_cap=evidence_cap,
            comparative_query=comparative_query
            or balanced_query
            or intent.has("temporal_delta")
            or intent.has("outlook_scope"),
        )
        selected = _ensure_temporal_delta_evidence(
            intent=intent,
            selected_nodes=selected,
            ranked_nodes=ranked_nodes,
            evidence_cap=evidence_cap,
        )
        selected = _ensure_named_entity_evidence(
            question=question,
            intent=intent,
            selected_nodes=selected,
            ranked_nodes=ranked_nodes,
            evidence_cap=evidence_cap,
        )
        selected = _ensure_outlook_scope_evidence(
            intent=intent,
            selected_nodes=selected,
            ranked_nodes=ranked_nodes,
            evidence_cap=evidence_cap,
        )
        selected = _ensure_metric_variant_evidence(
            question=question,
            intent=intent,
            selected_nodes=selected,
            ranked_nodes=ranked_nodes,
            evidence_cap=evidence_cap,
        )
        selected = _ensure_structured_evidence(
            question=question,
            selected_nodes=selected,
            ranked_nodes=ranked_nodes,
            evidence_cap=evidence_cap,
        )
        selected = _ensure_direct_metric_evidence(
            question=question,
            selected_nodes=selected,
            ranked_nodes=ranked_nodes,
            evidence_cap=evidence_cap,
        )

        selected = _ensure_image_evidence(
            question=question,
            selected_nodes=selected,
            ranked_nodes=ranked_nodes,
            evidence_cap=evidence_cap,
        )

        return selected

    def _resolve_evidence_cap(
        self,
        *,
        comparative_query: bool,
        conflict_focused_query: bool,
    ) -> int:
        if conflict_focused_query:
            return self.conflict_evidence_limit
        if comparative_query:
            return self.comparative_evidence_limit
        return self.evidence_limit

    def _resolve_rank_top_k(
        self,
        *,
        comparative_query: bool,
        conflict_focused_query: bool,
        high_diversity_query: bool = False,
    ) -> int:
        if conflict_focused_query:
            return max(self.top_k + 8, self.conflict_evidence_limit)
        if high_diversity_query:
            return max(self.top_k + 25, self.comparative_evidence_limit * 2)
        if comparative_query:
            return max(self.top_k + 8, self.comparative_evidence_limit)
        return self.top_k

    def _prioritize_conflict_candidates(
        self,
        *,
        ranked_nodes: list,
        conflict_focused_query: bool,
    ) -> list:
        if not conflict_focused_query:
            return ranked_nodes

        # For conflict queries, prioritize chunks that actually carry numeric signals.
        numeric_nodes = [node for node in ranked_nodes if _has_numeric_signal(node)]
        non_numeric_nodes = [
            node for node in ranked_nodes if not _has_numeric_signal(node)
        ]
        return numeric_nodes + non_numeric_nodes

    def _partition_reasoning_candidates(
        self,
        *,
        question: str,
        candidate_nodes: list,
    ) -> tuple[list, list]:
        if _is_reasoning_priority_query(question):
            return candidate_nodes, []

        primary_candidates = [
            node for node in candidate_nodes if not _is_reasoning_chunk(node)
        ]
        secondary_candidates = [
            node for node in candidate_nodes if _is_reasoning_chunk(node)
        ]
        return primary_candidates, secondary_candidates

    def _collect_evidence_candidates(
        self,
        *,
        primary_candidates: list,
        secondary_candidates: list,
        evidence_cap: int,
        comparative_query: bool,
    ) -> list:
        selected: list[Any] = []
        selected_keys: set[str] = set()
        candidate_batches = [primary_candidates]
        if secondary_candidates:
            candidate_batches.append(secondary_candidates)

        if comparative_query:
            selected_diversity_keys: set[str] = set()
            if self._collect_diverse_nodes(
                candidate_batches=candidate_batches,
                selected=selected,
                selected_keys=selected_keys,
                selected_diversity_keys=selected_diversity_keys,
                evidence_cap=evidence_cap,
            ):
                return selected

        self._collect_unique_nodes(
            candidate_batches=candidate_batches,
            selected=selected,
            selected_keys=selected_keys,
            evidence_cap=evidence_cap,
        )
        return selected

    def _collect_diverse_nodes(
        self,
        *,
        candidate_batches: list[list[Any]],
        selected: list[Any],
        selected_keys: set[str],
        selected_diversity_keys: set[str],
        evidence_cap: int,
    ) -> bool:
        # Prefer source diversity first: version label when available, otherwise
        # document family/document ID as fallback.
        for batch in candidate_batches:
            for node in batch:
                diversity_key = _evidence_diversity_key(node)
                if diversity_key in selected_diversity_keys:
                    continue
                key = _node_unique_key(node)
                if key in selected_keys:
                    continue
                selected.append(node)
                selected_keys.add(key)
                selected_diversity_keys.add(diversity_key)
                if len(selected) >= evidence_cap:
                    return True
        return False

    def _collect_unique_nodes(
        self,
        *,
        candidate_batches: list[list[Any]],
        selected: list[Any],
        selected_keys: set[str],
        evidence_cap: int,
    ) -> None:
        for batch in candidate_batches:
            for node in batch:
                if len(selected) >= evidence_cap:
                    return
                key = _node_unique_key(node)
                if key in selected_keys:
                    continue
                selected.append(node)
                selected_keys.add(key)

    def _synthesize_answer(
        self,
        question: str,
        citations: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.answer_synthesizer.synthesize(
            question=question,
            citations=citations,
            conflicts=conflicts,
        )


def _fuse_node_batches(node_batches: list[list]) -> list:
    fused: dict[str, tuple[float, Any]] = {}

    for batch in node_batches:
        for rank, node in enumerate(batch, start=1):
            key = _node_fusion_key(node, rank)
            retrieval_score, fused_score = _initial_fused_score(node, rank)

            existing = fused.get(key)
            if existing:
                fused_score += existing[0]
                node = _choose_higher_scored_node(existing[1], node, retrieval_score)

            node.score = fused_score
            fused[key] = (fused_score, node)

    return [
        node
        for _, node in sorted(fused.values(), key=lambda item: item[0], reverse=True)
    ]


def _retrieve_query_batches(base_retriever: Any, queries: list[str]) -> list[list[Any]]:
    return [base_retriever.retrieve(query) for query in queries]


def _dedupe_retrieval_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = " ".join(str(query or "").split())
        normalized = cleaned.lower()
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(cleaned)
    return deduped


def _empty_retrieval_intent_metadata(intent: RetrievalIntent) -> dict[str, Any]:
    return _retrieval_intent_metadata(
        intent=intent,
        companion_queries=list(intent.companion_queries),
        companion_counts={},
    )


def _retrieval_intent_metadata(
    *,
    intent: RetrievalIntent,
    companion_queries: list[str],
    companion_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "intent_labels": list(intent.labels),
        "companion_queries": companion_queries,
        "companion_counts_by_query": companion_counts,
    }


def _node_fusion_key(node: Any, rank: int) -> str:
    node_id = getattr(node.node, "node_id", None)
    metadata = node.node.metadata or {}
    key = node_id or metadata.get("chunk_id") or metadata.get("document_id")
    if key:
        return str(key)
    return f"{metadata.get('file_name', 'unknown')}:{rank}:{node.node.text[:64]}"


def _initial_fused_score(node: Any, rank: int) -> tuple[float, float]:
    retrieval_score = float(node.score or 0.0)
    rrf_score = 1.0 / (60 + rank)
    return retrieval_score, retrieval_score + rrf_score


def _choose_higher_scored_node(
    existing_node: Any, candidate_node: Any, candidate_retrieval_score: float
) -> Any:
    return (
        existing_node
        if (existing_node.score or 0.0) >= candidate_retrieval_score
        else candidate_node
    )


def _is_comparison_or_conflict_query(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in COMPARATIVE_TERMS)


def _needs_balanced_evidence_query(question: str) -> bool:
    normalized = question.lower()
    intent = analyze_retrieval_intent(question)
    if any(term in normalized for term in BROAD_BALANCED_TERMS):
        return True
    return len(intent.entities) >= 2


def _is_high_diversity_intent(intent: RetrievalIntent) -> bool:
    return any(
        intent.has(label)
        for label in (
            "temporal_delta",
            "outlook_scope",
            "named_entity_comparison",
        )
    )


def _is_conflict_focused_query(question: str) -> bool:
    normalized = question.lower()
    conflict_terms = (
        "conflict",
        "conflicting",
        "contradict",
        "contradiction",
        "across documents",
        "across sources",
        "disagree",
    )
    return any(term in normalized for term in conflict_terms)


def _has_numeric_signal(node: Any) -> bool:
    metadata = node.node.metadata or {}
    raw = metadata.get("contains_numeric_data")
    return safe_bool(raw, False)


def _node_has_image_assets(node: Any) -> bool:
    metadata = node.node.metadata or {}
    refs = metadata.get("asset_refs")
    if isinstance(refs, str):
        return bool(refs.strip())
    if isinstance(refs, list):
        return any(isinstance(ref, str) and ref.strip() for ref in refs)
    return False


def _is_reasoning_chunk(node: Any) -> bool:
    metadata = node.node.metadata or {}
    return str(metadata.get("chunk_type") or "") in REASONING_CHUNK_TYPES


def _is_reasoning_priority_query(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in REASONING_PRIORITY_TERMS)


def _is_visual_or_image_query(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in VISUAL_QUERY_TERMS)


def _is_time_anchored_query(question: str) -> bool:
    normalized = question.lower()
    if YEAR_PATTERN.search(normalized):
        return True
    return any(term in normalized for term in TIME_ANCHORED_TERMS)


def _wants_table_evidence(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in TABLE_EVIDENCE_TERMS)


def _wants_chart_evidence(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in CHART_EVIDENCE_TERMS)


def _is_table_like_node(node: Any) -> bool:
    metadata = node.node.metadata or {}
    chunk_type = str(metadata.get("chunk_type") or "")
    return chunk_type in {
        "full_table",
        "table_segment",
        "table_summary_text",
        "reasoning_table",
    } or safe_bool(metadata.get("table_detected"), False)


def _is_chart_like_node(node: Any) -> bool:
    metadata = node.node.metadata or {}
    chunk_type = str(metadata.get("chunk_type") or "")
    figure_type = str(metadata.get("figure_type") or "")
    return (
        chunk_type
        in {
            "figure_artifact",
            "chart_context",
            "chart_data_points",
            "visual_proxy_text",
            "reasoning_chart",
            "reasoning_figure",
        }
        or safe_bool(metadata.get("chart_detected"), False)
        or figure_type in {"chart", "diagram", "infographic"}
    )


def _ensure_structured_evidence(
    *,
    question: str,
    selected_nodes: list[Any],
    ranked_nodes: list[Any],
    evidence_cap: int,
) -> list[Any]:
    want_table = _wants_table_evidence(question)
    want_chart = _wants_chart_evidence(question)
    if not want_table and not want_chart:
        return selected_nodes

    selected = list(selected_nodes)
    selected_keys = {_node_unique_key(node) for node in selected}
    requirements: list[tuple[bool, Any, str]] = [
        (want_table, _is_table_like_node, "table"),
        (want_chart, _is_chart_like_node, "chart"),
    ]

    for enabled, predicate, requirement_name in requirements:
        if not enabled:
            continue
        if any(predicate(node) for node in selected):
            continue

        candidate = next(
            (
                node
                for node in ranked_nodes
                if predicate(node) and _node_unique_key(node) not in selected_keys
            ),
            None,
        )
        if candidate is None:
            logger.debug(
                "Structured evidence requirement unmet (type=%s): no ranked candidate",
                requirement_name,
            )
            continue

        if len(selected) < evidence_cap:
            selected.append(candidate)
            selected_keys.add(_node_unique_key(candidate))
            continue

        replace_idx = next(
            (
                idx
                for idx in range(len(selected) - 1, -1, -1)
                if not predicate(selected[idx])
            ),
            None,
        )
        if replace_idx is None:
            continue

        selected_keys.discard(_node_unique_key(selected[replace_idx]))
        selected[replace_idx] = candidate
        selected_keys.add(_node_unique_key(candidate))

    return selected


def _ensure_image_evidence(
    *,
    question: str,
    selected_nodes: list[Any],
    ranked_nodes: list[Any],
    evidence_cap: int,
) -> list[Any]:
    if not _is_visual_or_image_query(question):
        return selected_nodes

    desired_image_nodes = min(2, evidence_cap)
    selected = list(selected_nodes)
    selected_keys = {_node_unique_key(node) for node in selected}
    image_node_count = sum(1 for node in selected if _node_has_image_assets(node))

    for candidate in ranked_nodes:
        if image_node_count >= desired_image_nodes:
            break
        if not _node_has_image_assets(candidate):
            continue
        candidate_key = _node_unique_key(candidate)
        if candidate_key in selected_keys:
            continue

        if len(selected) < evidence_cap:
            selected.append(candidate)
            selected_keys.add(candidate_key)
            image_node_count += 1
            continue

        replace_idx = next(
            (
                idx
                for idx in range(len(selected) - 1, -1, -1)
                if not _node_has_image_assets(selected[idx])
            ),
            None,
        )
        if replace_idx is None:
            break

        selected_keys.discard(_node_unique_key(selected[replace_idx]))
        selected[replace_idx] = candidate
        selected_keys.add(candidate_key)
        image_node_count += 1

    return selected


METRIC_VALUE_PATTERN = re.compile(
    r"\b\$?\d[\d,]*(?:\.\d+)?\s*(?:[+%x]|bn|m|billion|million)?\s+"
    r"(?:global\s+)?(?P<subject>[a-z][a-z-]+)s?\b",
    re.IGNORECASE,
)
METRIC_STOPWORDS = {
    "does",
    "have",
    "has",
    "many",
    "much",
    "what",
    "which",
    "with",
    "from",
    "that",
    "this",
    "digital",
    "realty",
}


def _ensure_direct_metric_evidence(
    *,
    question: str,
    selected_nodes: list[Any],
    ranked_nodes: list[Any],
    evidence_cap: int,
) -> list[Any]:
    normalized_question = question.lower()
    if not any(
        term in normalized_question for term in ("how many", "number of", "count")
    ):
        return selected_nodes

    candidates = [
        node
        for node in ranked_nodes
        if _is_direct_metric_node(node, normalized_question)
    ]
    if not candidates:
        return selected_nodes

    best = max(candidates, key=_direct_metric_priority)
    best_key = _node_unique_key(best)
    selected = list(selected_nodes)
    selected_keys = {_node_unique_key(node) for node in selected}
    if best_key in selected_keys:
        return [best] + [
            node for node in selected if _node_unique_key(node) != best_key
        ]

    if len(selected) < evidence_cap:
        selected.insert(0, best)
        return selected

    replace_idx = next(
        (
            idx
            for idx in range(len(selected) - 1, -1, -1)
            if not _is_direct_metric_node(selected[idx], normalized_question)
        ),
        None,
    )
    if replace_idx is None:
        return selected
    selected[replace_idx] = best
    return selected


def _ensure_temporal_delta_evidence(
    *,
    intent: RetrievalIntent,
    selected_nodes: list[Any],
    ranked_nodes: list[Any],
    evidence_cap: int,
) -> list[Any]:
    if not intent.has("temporal_delta"):
        return selected_nodes
    return _ensure_group_quota(
        selected_nodes=selected_nodes,
        ranked_nodes=ranked_nodes,
        evidence_cap=evidence_cap,
        group_key_fn=_document_or_version_group_key,
        target_groups=_top_groups(
            ranked_nodes, _document_or_version_group_key, limit=2
        ),
        quota_per_group=3,
    )


def _ensure_named_entity_evidence(
    *,
    question: str,
    intent: RetrievalIntent,
    selected_nodes: list[Any],
    ranked_nodes: list[Any],
    evidence_cap: int,
) -> list[Any]:
    aliases_by_entity = dict(intent.entities) or _query_entity_aliases(question.lower())
    if len(aliases_by_entity) < 2:
        return selected_nodes

    quota = min(3, max(1, evidence_cap // max(len(aliases_by_entity), 1)))
    selected = list(selected_nodes)
    selected_keys = {_node_unique_key(node) for node in selected}

    for aliases in aliases_by_entity.values():
        while (
            sum(1 for node in selected if _node_matches_entity(node, aliases)) < quota
        ):
            candidate = next(
                (
                    node
                    for node in ranked_nodes
                    if _node_unique_key(node) not in selected_keys
                    and _node_matches_entity(node, aliases)
                ),
                None,
            )
            if candidate is None:
                break

            candidate_key = _node_unique_key(candidate)
            if len(selected) < evidence_cap:
                selected.append(candidate)
                selected_keys.add(candidate_key)
                continue

            current_count = sum(
                1 for node in selected if _node_matches_entity(node, aliases)
            )
            replace_idx = _least_useful_entity_replacement_index(
                selected=selected,
                required_aliases=aliases_by_entity.values(),
                target_aliases=aliases,
            )
            if replace_idx is None:
                break
            selected_keys.discard(_node_unique_key(selected[replace_idx]))
            selected[replace_idx] = candidate
            selected_keys.add(candidate_key)
            if (
                sum(1 for node in selected if _node_matches_entity(node, aliases))
                <= current_count
            ):
                break

    return selected


def _query_entity_aliases(normalized_question: str) -> dict[str, tuple[str, ...]]:
    return query_entity_aliases(normalized_question)


def _ensure_outlook_scope_evidence(
    *,
    intent: RetrievalIntent,
    selected_nodes: list[Any],
    ranked_nodes: list[Any],
    evidence_cap: int,
) -> list[Any]:
    if not intent.has("outlook_scope"):
        return selected_nodes

    selected = list(selected_nodes)
    selected_keys = {_node_unique_key(node) for node in selected}
    desired_scope_nodes = 3
    for candidate in ranked_nodes:
        if (
            sum(
                1
                for node in selected
                if _node_has_scope_terms(node, MERGER_SCOPE_TERMS)
            )
            >= desired_scope_nodes
        ):
            return selected
        if not _node_has_scope_terms(candidate, MERGER_SCOPE_TERMS):
            continue
        candidate_key = _node_unique_key(candidate)
        if candidate_key in selected_keys:
            continue
        _append_or_replace_node(
            selected=selected,
            selected_keys=selected_keys,
            candidate=candidate,
            evidence_cap=evidence_cap,
            replace_predicate=lambda node: not _node_has_scope_terms(
                node, MERGER_SCOPE_TERMS
            ),
        )
    return selected


def _ensure_metric_variant_evidence(
    *,
    question: str,
    intent: RetrievalIntent,
    selected_nodes: list[Any],
    ranked_nodes: list[Any],
    evidence_cap: int,
) -> list[Any]:
    if not intent.has("caveat_inconsistency"):
        return selected_nodes

    normalized_question = question.lower()
    subjects = _metric_query_subjects(normalized_question)
    if not subjects:
        return selected_nodes

    selected = list(selected_nodes)
    selected_keys = {_node_unique_key(node) for node in selected}
    selected_values = _metric_values_for_nodes(selected, subjects)
    for candidate in ranked_nodes:
        candidate_values = _metric_values_for_node(candidate, subjects)
        missing_values = candidate_values - selected_values
        if not missing_values:
            continue
        candidate_key = _node_unique_key(candidate)
        if candidate_key in selected_keys:
            selected_values.update(candidate_values)
            continue
        _append_or_replace_node(
            selected=selected,
            selected_keys=selected_keys,
            candidate=candidate,
            evidence_cap=evidence_cap,
            replace_predicate=lambda node: not _metric_values_for_node(node, subjects),
        )
        selected_values.update(candidate_values)
        if len(selected_values) >= 2:
            return selected
    return selected


MERGER_SCOPE_TERMS = (
    "merger",
    "pro forma",
    "pro-forma",
    "acquisition",
    "neutral",
    "accretive",
    "stabilization",
    "combined",
)


def _node_has_scope_terms(node: Any, terms: tuple[str, ...]) -> bool:
    text = _node_search_text(node)
    return any(term in text for term in terms)


def _metric_values_for_nodes(nodes: list[Any], subjects: set[str]) -> set[str]:
    values: set[str] = set()
    for node in nodes:
        values.update(_metric_values_for_node(node, subjects))
    return values


def _metric_values_for_node(node: Any, subjects: set[str]) -> set[str]:
    values: set[str] = set()
    text = node.node.text or ""
    for match in METRIC_VALUE_PATTERN.finditer(text):
        if normalize_metric_subject(match.group("subject")) in subjects:
            values.add(_normalize_metric_value(match.group(0)))
    return values


def _normalize_metric_value(raw_value: str) -> str:
    return " ".join(raw_value.lower().replace(",", "").split())


def _ensure_group_quota(
    *,
    selected_nodes: list[Any],
    ranked_nodes: list[Any],
    evidence_cap: int,
    group_key_fn: Any,
    target_groups: list[str],
    quota_per_group: int,
) -> list[Any]:
    if len(target_groups) < 2:
        return selected_nodes

    selected = list(selected_nodes)
    selected_keys = {_node_unique_key(node) for node in selected}
    for group in target_groups:
        while _count_group(selected, group_key_fn, group) < quota_per_group:
            candidate = next(
                (
                    node
                    for node in ranked_nodes
                    if _node_unique_key(node) not in selected_keys
                    and group_key_fn(node) == group
                ),
                None,
            )
            if candidate is None:
                break
            _append_or_replace_node(
                selected=selected,
                selected_keys=selected_keys,
                candidate=candidate,
                evidence_cap=evidence_cap,
                replace_predicate=lambda node: group_key_fn(node) not in target_groups
                or _count_group(selected, group_key_fn, group_key_fn(node))
                > quota_per_group,
            )
    return selected


def _top_groups(nodes: list[Any], group_key_fn: Any, limit: int) -> list[str]:
    groups: list[str] = []
    for node in nodes:
        group = group_key_fn(node)
        if group and group not in groups:
            groups.append(group)
        if len(groups) >= limit:
            break
    return groups


def _count_group(nodes: list[Any], group_key_fn: Any, group: str) -> int:
    return sum(1 for node in nodes if group_key_fn(node) == group)


def _document_or_version_group_key(node: Any) -> str:
    metadata = node.node.metadata or {}
    return str(
        metadata.get("document_id")
        or metadata.get("document_name")
        or metadata.get("source_file")
        or metadata.get("version_label")
        or metadata.get("document_version_group")
        or _node_unique_key(node)
    )


def _append_or_replace_node(
    *,
    selected: list[Any],
    selected_keys: set[str],
    candidate: Any,
    evidence_cap: int,
    replace_predicate: Any,
) -> None:
    candidate_key = _node_unique_key(candidate)
    if candidate_key in selected_keys:
        return
    if len(selected) < evidence_cap:
        selected.append(candidate)
        selected_keys.add(candidate_key)
        return

    replace_idx = next(
        (
            idx
            for idx in range(len(selected) - 1, -1, -1)
            if replace_predicate(selected[idx])
        ),
        None,
    )
    if replace_idx is None:
        replace_idx = len(selected) - 1
    selected_keys.discard(_node_unique_key(selected[replace_idx]))
    selected[replace_idx] = candidate
    selected_keys.add(candidate_key)


def _node_matches_entity(node: Any, aliases: tuple[str, ...]) -> bool:
    haystack = _node_search_text(node)
    return any(alias in haystack for alias in aliases)


def _node_search_text(node: Any) -> str:
    metadata = node.node.metadata or {}
    return " ".join(
        str(value or "")
        for value in (
            metadata.get("document_name"),
            metadata.get("file_name"),
            metadata.get("source_file"),
            metadata.get("document_family"),
            metadata.get("section_path"),
            metadata.get("section_title"),
            metadata.get("llm_page_summary"),
            metadata.get("llm_caption"),
            node.node.text,
        )
    ).lower()


def _least_useful_entity_replacement_index(
    *,
    selected: list[Any],
    required_aliases: Any,
    target_aliases: tuple[str, ...] | None = None,
) -> int | None:
    for idx in range(len(selected) - 1, -1, -1):
        if target_aliases and _node_matches_entity(selected[idx], target_aliases):
            continue
        if not any(
            _node_matches_entity(selected[idx], aliases) for aliases in required_aliases
        ):
            return idx
    for idx in range(len(selected) - 1, -1, -1):
        if target_aliases and _node_matches_entity(selected[idx], target_aliases):
            continue
        return idx
    return None


def _is_direct_metric_node(node: Any, normalized_question: str) -> bool:
    subjects = _metric_query_subjects(normalized_question)
    if not subjects:
        return False
    text = (node.node.text or "").lower()
    return any(
        normalize_metric_subject(match.group("subject")) in subjects
        for match in METRIC_VALUE_PATTERN.finditer(text)
    )


def _direct_metric_priority(node: Any) -> tuple[int, int, float]:
    metadata = node.node.metadata or {}
    chunk_type = str(metadata.get("chunk_type") or "")
    return (
        safe_int(metadata.get("version_rank"), 0),
        1 if chunk_type == "body_text" else 0,
        float(node.score or 0.0),
    )


def _metric_query_subjects(normalized_query: str) -> set[str]:
    subjects: set[str] = set()
    for token in re.findall(r"[a-z][a-z-]+", normalized_query):
        if len(token) < 4 or token in METRIC_STOPWORDS:
            continue
        subjects.add(normalize_metric_subject(token))
    return subjects


def _build_retrieval_diagnostics(
    ranked_nodes: list[Any],
    evidence_nodes: list[Any],
    intent: RetrievalIntent | None = None,
) -> dict[str, Any]:
    ranked_chunk_types = _chunk_type_counts(ranked_nodes)
    evidence_chunk_types = _chunk_type_counts(evidence_nodes)
    ranked_reasoning = sum(
        count
        for chunk_type, count in ranked_chunk_types.items()
        if chunk_type in REASONING_CHUNK_TYPES
    )
    evidence_reasoning = sum(
        count
        for chunk_type, count in evidence_chunk_types.items()
        if chunk_type in REASONING_CHUNK_TYPES
    )
    ranked_image_count = sum(1 for node in ranked_nodes if _node_has_image_assets(node))
    evidence_image_count = sum(
        1 for node in evidence_nodes if _node_has_image_assets(node)
    )

    return {
        "ranked_document_count": len(_document_diversity_keys(ranked_nodes)),
        "evidence_document_count": len(_document_diversity_keys(evidence_nodes)),
        "ranked_chunk_types": ranked_chunk_types,
        "evidence_chunk_types": evidence_chunk_types,
        "ranked_reasoning_count": ranked_reasoning,
        "evidence_reasoning_count": evidence_reasoning,
        "ranked_image_chunk_count": ranked_image_count,
        "evidence_image_chunk_count": evidence_image_count,
        "intent_labels": list(intent.labels) if intent else [],
        "evidence_by_document": _node_counts_by_document(evidence_nodes),
        "evidence_by_version_group": _node_counts_by_version_group(evidence_nodes),
        "evidence_by_entity": _node_counts_by_entity(evidence_nodes, intent),
    }


def _chunk_type_counts(nodes: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        metadata = node.node.metadata or {}
        chunk_type = str(metadata.get("chunk_type") or "unknown")
        counts[chunk_type] = counts.get(chunk_type, 0) + 1
    return counts


def _document_diversity_keys(nodes: list[Any]) -> set[str]:
    keys: set[str] = set()
    for node in nodes:
        metadata = node.node.metadata or {}
        value = (
            metadata.get("document_id")
            or metadata.get("document_name")
            or metadata.get("source_file")
            or getattr(node.node, "node_id", None)
        )
        if value:
            keys.add(str(value))
    return keys


def _node_counts_by_document(nodes: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        metadata = node.node.metadata or {}
        label = str(
            metadata.get("document_name")
            or metadata.get("source_file")
            or metadata.get("document_id")
            or "unknown"
        )
        counts[label] = counts.get(label, 0) + 1
    return counts


def _node_counts_by_version_group(nodes: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        metadata = node.node.metadata or {}
        label = str(
            metadata.get("version_label")
            or metadata.get("document_version_group")
            or metadata.get("document_family")
            or metadata.get("document_name")
            or metadata.get("document_id")
            or "unknown"
        )
        counts[label] = counts.get(label, 0) + 1
    return counts


def _node_counts_by_entity(
    nodes: list[Any], intent: RetrievalIntent | None
) -> dict[str, int]:
    if intent is None or not intent.entities:
        return {}
    counts: dict[str, int] = {}
    for entity_name, aliases in intent.entities.items():
        count = sum(1 for node in nodes if _node_matches_entity(node, aliases))
        if count:
            counts[entity_name] = count
    return counts


def _evidence_diversity_key(node: Any) -> str:
    metadata = node.node.metadata or {}
    version_label = (metadata.get("version_label") or "").strip().lower()
    if version_label:
        return f"version:{version_label}"

    version_group = metadata.get("document_version_group") or metadata.get(
        "document_family"
    )
    if version_group:
        return f"group:{str(version_group).strip().lower()}"

    document_id = metadata.get("document_id")
    if document_id:
        return f"doc:{document_id}"

    doc_name = metadata.get("document_name") or metadata.get("source_file")
    if doc_name:
        return f"name:{str(doc_name).strip().lower()}"

    return f"node:{_node_unique_key(node)}"


def _node_unique_key(node: Any) -> str:
    metadata = node.node.metadata or {}
    return str(
        getattr(node.node, "node_id", None)
        or metadata.get("chunk_id")
        or metadata.get("document_id")
        or metadata.get("citation_label")
        or node.node.text[:80]
    )


def _build_grounded_prompt(
    question: str,
    citations: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    image_attachment_count: int = 0,
    conversation_context: dict[str, Any] | None = None,
) -> str:
    evidence_lines = []
    for index, citation in enumerate(citations, start=1):
        label = _citation_label(citation, index)
        version = citation.get("version_label") or "unknown"
        chunk_type = citation.get("chunk_type") or "text"
        location = _citation_location(citation)
        source_context = _citation_source_context(citation)
        image_assets = citation.get("asset_refs") or []
        enriched = _citation_enriched_facts(citation)
        excerpt = _prompt_excerpt(citation)
        enriched_block = f"Extracted facts:\n{enriched}\n" if enriched else ""
        evidence_lines.append(
            f"[{index}] {label} | version={version} | chunk_type={chunk_type}"
            f"{location} | image_assets={len(image_assets)}\n"
            f"{source_context}"
            f"{enriched_block}Excerpt:\n{excerpt}"
        )

    conflict_lines = []
    for conflict in conflicts[:2]:
        summary = (conflict.get("summary") or "").strip()
        if summary:
            conflict_lines.append(f"- {summary}")

    conflict_block = (
        "\n".join(conflict_lines)
        if conflict_lines
        else (
            "- No high-confidence conflicts were detected in the selected excerpts. "
            "This is not proof that all documents are conflict-free."
        )
    )
    evidence_block = "\n".join(evidence_lines)
    conversation_block = _build_conversation_context_block(conversation_context or {})

    return build_grounded_answer_prompt(
        question=question,
        image_attachment_count=image_attachment_count,
        evidence_block=evidence_block,
        conflict_block=conflict_block,
        conversation_context_block=conversation_block,
        answering_notes_block=_build_answering_notes(question, citations),
    )


def _build_labeled_context_sections(
    *,
    question: str,
    citations: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    conversation_context: dict[str, Any],
) -> dict[str, Any]:
    summary = str(conversation_context.get("session_summary") or "").strip()
    cross_session_pairs = conversation_context.get("cross_session_pairs") or []

    return {
        "session_summary": summary,
        "cross_session_lines": _build_context_cross_session_lines(cross_session_pairs),
        "evidence_lines": _build_context_evidence_lines(citations),
        "conflict_lines": [
            *_build_answering_notes(question, citations).splitlines(),
            *_build_context_conflict_lines(conflicts),
        ],
    }


def _build_conversation_context_block(conversation_context: dict[str, Any]) -> str:
    summary = str(conversation_context.get("session_summary") or "").strip()
    recent_turns = conversation_context.get("recent_turns") or []
    cross_session_pairs = conversation_context.get("cross_session_pairs") or []

    sections: list[str] = []
    if summary:
        sections.append(f"SESSION_SUMMARY:\n{summary}")

    if recent_turns:
        lines = _format_recent_turn_lines(recent_turns)
        if lines:
            sections.append("CURRENT_SESSION_RECENT_TURNS:\n" + "\n".join(lines))

    if cross_session_pairs:
        lines = _format_cross_session_lines(cross_session_pairs)
        if lines:
            sections.append("CROSS_SESSION_RELEVANT_QA:\n" + "\n".join(lines))

    if not sections:
        return "NO_PRIOR_CONVERSATION_CONTEXT"
    return "\n\n".join(sections)


def _build_context_evidence_lines(citations: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, citation in enumerate(citations, start=1):
        label = _citation_label(citation, index)
        version = citation.get("version_label") or "unknown"
        chunk_type = citation.get("chunk_type") or "text"
        location = _citation_location(citation)
        source_context = _citation_source_context(citation)
        enriched = _citation_enriched_facts(citation)
        excerpt = _prompt_excerpt(citation)
        enriched_block = f"Extracted facts:\n{enriched}\n" if enriched else ""
        lines.append(
            f"[{index}] {label} | version={version} | chunk_type={chunk_type}{location}\n"
            f"{source_context}"
            f"{enriched_block}Excerpt: {excerpt}"
        )
    return lines


def _build_context_conflict_lines(conflicts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for conflict in conflicts[:4]:
        summary_line = _normalize_inline_text(conflict.get("summary"))
        if summary_line:
            lines.append(f"- {summary_line}")
    if not lines:
        return ["- No high-confidence conflicts were detected in selected evidence."]
    return lines


def _build_answering_notes(question: str, citations: list[dict[str, Any]]) -> str:
    intent = analyze_retrieval_intent(question)
    notes: list[str] = []

    if intent.has("temporal_delta"):
        notes.append(
            "- For this change question, organize the answer as older/baseline evidence, "
            "newer/update evidence, then stable vs changed or newly emphasized items."
        )

    if intent.has("stale_source") or _citations_have_source_dates(citations):
        notes.append(
            "- State absolute source/date scope for dated evidence; if figures describe "
            "an older data year, do not present them as current."
        )

    if intent.has("outlook_scope") and _citations_have_scope_terms(citations):
        notes.append(
            "- Separate standalone guidance from merger/acquisition/pro-forma, neutral, "
            "accretive, stabilization, or combined-company context."
        )

    if intent.has("caveat_inconsistency"):
        value_sources = _metric_value_sources_for_citations(question, citations)
        if len(value_sources) >= 2:
            source_summary = "; ".join(
                f"{value}: {', '.join(source_refs[:3])}"
                for value, source_refs in list(value_sources.items())[:5]
            )
            notes.append(
                "- Selected evidence contains multiple values for the same metric "
                f"({source_summary}); explicitly reconcile all values with "
                "document/date/page qualifiers, including same-document inconsistencies."
            )

    if intent.has("named_entity_comparison"):
        notes.append(
            "- Compare each named entity with the evidence available for that entity; "
            "partial scope evidence should be used with a precise limitation."
        )

    return "\n".join(notes)


def _citations_have_source_dates(citations: list[dict[str, Any]]) -> bool:
    for citation in citations:
        if citation.get("document_date") or citation.get("as_of_date"):
            return True
        enriched = citation.get("enriched_metadata")
        if isinstance(enriched, dict) and (
            enriched.get("document_date") or enriched.get("as_of_date")
        ):
            return True
    return False


def _citations_have_scope_terms(citations: list[dict[str, Any]]) -> bool:
    return any(
        any(term in _citation_search_text(citation) for term in MERGER_SCOPE_TERMS)
        for citation in citations
    )


def _metric_values_for_citations(
    question: str, citations: list[dict[str, Any]]
) -> set[str]:
    return set(_metric_value_sources_for_citations(question, citations))


def _metric_value_sources_for_citations(
    question: str, citations: list[dict[str, Any]]
) -> dict[str, list[str]]:
    subjects = _metric_query_subjects(question.lower())
    if not subjects:
        return {}

    values: dict[str, list[str]] = {}
    for index, citation in enumerate(citations, start=1):
        for match in METRIC_VALUE_PATTERN.finditer(_citation_search_text(citation)):
            if normalize_metric_subject(match.group("subject")) in subjects:
                value = _normalize_metric_value(match.group(0))
                values.setdefault(value, []).append(f"[{index}]")
    return values


def _citation_search_text(citation: dict[str, Any]) -> str:
    enriched = citation.get("enriched_metadata")
    enriched_text = ""
    if isinstance(enriched, dict):
        enriched_text = format_enriched_metadata_for_prompt(enriched)
    return " ".join(
        str(value or "")
        for value in (
            citation.get("document_name"),
            citation.get("citation_label"),
            citation.get("section_title"),
            citation.get("metric_basis"),
            citation.get("text"),
            enriched_text,
        )
    ).lower()


def _build_context_cross_session_lines(
    cross_session_pairs: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    for pair in cross_session_pairs:
        user_text = _normalize_inline_text(pair.get("user_text"))
        assistant_text = _normalize_inline_text(pair.get("assistant_text"))
        if not user_text or not assistant_text:
            continue
        score = pair.get("score")
        score_label = (
            f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        )
        lines.append(
            f"- similarity={score_label} | prior_user={user_text} | prior_assistant={assistant_text}"
        )
    return lines


def _normalize_inline_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _citation_label(citation: dict[str, Any], index: int) -> str:
    return citation.get("citation_label") or citation.get(
        "document_name", f"Source {index}"
    )


def _citation_location(citation: dict[str, Any]) -> str:
    if citation.get("page_num"):
        return f" | page={citation['page_num']}"
    if citation.get("slide_num"):
        return f" | slide={citation['slide_num']}"
    return ""


def _citation_source_context(citation: dict[str, Any]) -> str:
    parts: list[str] = []
    for label, key in (
        ("document_date", "document_date"),
        ("as_of_date", "as_of_date"),
        ("metric_basis", "metric_basis"),
        ("version_group", "version_group"),
        ("effective_from", "effective_from"),
        ("effective_to", "effective_to"),
        ("section", "section_title"),
    ):
        value = citation.get(key)
        if value:
            parts.append(f"{label}={value}")
    enriched = citation.get("enriched_metadata")
    if isinstance(enriched, dict):
        for label in ("document_date", "as_of_date", "metric_basis", "units"):
            if label not in {part.split("=", 1)[0] for part in parts} and enriched.get(
                label
            ):
                parts.append(f"{label}={enriched[label]}")
    if not parts:
        return ""
    return "Source context: " + " | ".join(str(part) for part in parts) + "\n"


def _truncate_with_ellipsis(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _format_recent_turn_lines(recent_turns: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for turn in recent_turns:
        role = str(turn.get("role") or "unknown").lower()
        role_label = "User" if role == "user" else "Assistant"
        content = _normalize_inline_text(turn.get("content"))
        if not content:
            continue
        lines.append(
            f"- {role_label}: {_truncate_with_ellipsis(content, RECENT_TURN_MAX_CHARS)}"
        )
    return lines


def _format_cross_session_lines(cross_session_pairs: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for pair in cross_session_pairs:
        user_text = _normalize_inline_text(pair.get("user_text"))
        assistant_text = _normalize_inline_text(pair.get("assistant_text"))
        if not user_text or not assistant_text:
            continue

        user_text = _truncate_with_ellipsis(user_text, CROSS_SESSION_USER_MAX_CHARS)
        assistant_text = _truncate_with_ellipsis(
            assistant_text, CROSS_SESSION_ASSISTANT_MAX_CHARS
        )
        score = pair.get("score")
        score_text = (
            f" (similarity={float(score):.3f})"
            if isinstance(score, (int, float))
            else ""
        )
        lines.append(f"- Prior Q{score_text}: {user_text}\n  Prior A: {assistant_text}")
    return lines


def _prompt_excerpt(citation: dict[str, Any]) -> str:
    chunk_type = str(citation.get("chunk_type") or "")
    rich_types = {
        "full_table",
        "table_segment",
        "table_summary_text",
        "figure_artifact",
        "chart_context",
        "chart_data_points",
        "visual_proxy_text",
    } | REASONING_CHUNK_TYPES
    limit = 2400 if chunk_type in rich_types else 1400
    text = (citation.get("text") or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n[...truncated]"


def _citation_enriched_facts(citation: dict[str, Any]) -> str:
    enriched = citation.get("enriched_metadata")
    if not isinstance(enriched, dict):
        return ""
    return format_enriched_metadata_for_prompt(enriched)


def _build_grounded_message(prompt: str, image_paths: list[str]) -> ChatMessage:
    blocks: list[Any] = [TextBlock(text=prompt)]
    for image_path in image_paths:
        mime_type = mimetypes.guess_type(image_path)[0]
        blocks.append(
            ImageBlock(
                path=image_path,
                image_mimetype=mime_type or "image/png",
            )
        )
    return ChatMessage(role=MessageRole.USER, blocks=blocks)


def _append_image_inputs(
    *,
    input_messages: list[dict[str, Any]],
    image_paths: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not image_paths:
        return input_messages, []

    if not input_messages:
        return input_messages, []

    user_message = input_messages[-1]
    if not isinstance(user_message, dict):
        return input_messages, []
    if user_message.get("role") != "user":
        return input_messages, []

    used_image_paths: list[str] = []
    image_inputs: list[dict[str, Any]] = []
    for path in image_paths:
        data_url = _image_path_to_data_url(path)
        if not data_url:
            continue
        image_inputs.append({"type": "input_image", "image_url": data_url})
        used_image_paths.append(path)

    if not image_inputs:
        return input_messages, []

    messages = [*input_messages]
    user_message = {**user_message}
    user_content = user_message.get("content")
    if isinstance(user_content, list):
        multimodal_content = [
            {**item} if isinstance(item, dict) else item for item in user_content
        ]
    else:
        multimodal_content = [{"type": "input_text", "text": str(user_content or "")}]

    multimodal_content.extend(image_inputs)
    user_message["content"] = multimodal_content
    messages[-1] = user_message
    return messages, used_image_paths


def _collect_image_evidence_paths(
    citations: list[dict[str, Any]], max_images: int = MAX_MULTIMODAL_IMAGES
) -> list[str]:
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    image_paths: list[str] = []

    for citation in citations:
        refs = citation.get("asset_refs") or []
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list):
            continue

        artifact_bundle_path = citation.get("artifact_bundle_path")
        for ref in refs:
            resolved = _resolve_asset_path(ref, artifact_bundle_path)
            if resolved is None:
                continue
            if resolved.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue

            path_str = str(resolved)
            if path_str in seen_paths:
                continue
            try:
                content_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
            except OSError:
                continue
            if content_hash in seen_hashes:
                continue

            seen_paths.add(path_str)
            seen_hashes.add(content_hash)
            image_paths.append(path_str)
            if len(image_paths) >= max_images:
                return image_paths

    return image_paths


def _resolve_asset_path(raw_ref: Any, artifact_bundle_path: Any) -> Path | None:
    if not isinstance(raw_ref, str):
        return None

    ref = raw_ref.strip()
    if not ref:
        return None

    candidates: list[Path] = []
    raw_path = Path(ref).expanduser()

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        if isinstance(artifact_bundle_path, str) and artifact_bundle_path.strip():
            candidates.append(Path(artifact_bundle_path).expanduser() / raw_path)
        candidates.append(raw_path)
        candidates.append(Path.cwd() / raw_path)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved

    return None


def _image_path_to_data_url(image_path: str) -> str | None:
    candidate = Path(image_path)
    if not candidate.exists() or not candidate.is_file():
        return None
    mime_type = mimetypes.guess_type(str(candidate))[0] or "image/png"
    try:
        encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:{mime_type};base64,{encoded}"


def _extract_answer_and_reasoning_from_chat(response: Any) -> tuple[str, str | None]:
    message = getattr(response, "message", None)
    if message is None:
        return _split_reasoning_from_text(str(response).strip())

    answer_parts: list[str] = []
    for block in getattr(message, "blocks", None) or []:
        if isinstance(block, TextBlock):
            content = (block.text or "").strip()
            if content:
                answer_parts.append(content)

    answer_text = "\n".join(answer_parts).strip()
    parsed_answer, parsed_reasoning = _split_reasoning_from_text(
        answer_text or str(getattr(message, "content", "") or "").strip()
    )
    reasoning_text = extract_chat_response_reasoning(response) or parsed_reasoning
    return parsed_answer, reasoning_text


def _split_reasoning_from_text(text: str) -> tuple[str, str | None]:
    if not text:
        return "", None

    text = text.strip()
    thinking_match = re.search(
        r"<thinking>(.*?)</thinking>", text, re.IGNORECASE | re.DOTALL
    )
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)

    reasoning = thinking_match.group(1).strip() if thinking_match else None

    if answer_match:
        answer = answer_match.group(1).strip()
    elif thinking_match:
        answer = re.sub(
            r"<thinking>.*?</thinking>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        answer = re.sub(r"</?answer>", "", answer, flags=re.IGNORECASE).strip()
    elif re.search(r"<thinking\b[^>]*>", text, re.IGNORECASE):
        open_answer_match = re.search(
            r"<answer\b[^>]*>(.*)$", text, re.IGNORECASE | re.DOTALL
        )
        if open_answer_match:
            answer = open_answer_match.group(1).strip()
        else:
            answer = re.sub(
                r"<thinking\b[^>]*>.*$",
                "",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
        answer = re.sub(r"</?answer>", "", answer, flags=re.IGNORECASE).strip()
    else:
        answer = text

    return answer, reasoning


def _build_source_grounded_fallback(citations: list[dict[str, Any]]) -> str:
    labels = []
    for citation in citations[:3]:
        label = citation.get("citation_label") or citation.get("document_name")
        if label:
            labels.append(label)

    if labels:
        return (
            "I found relevant evidence, but could not synthesize a final answer. "
            f"Review these sources: {', '.join(labels)}."
        )
    return "I found relevant evidence, but could not synthesize a final answer."
