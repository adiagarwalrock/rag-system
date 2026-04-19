"""
Retriever: orchestrates the full retrieval pipeline.

Flow: vector retrieval -> ranking -> evidence selection -> conflict checks -> citation building
"""

import logging
import mimetypes
import re
import hashlib
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
    ThinkingBlock,
)
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from app.core.ai_provider import (
    extract_chat_response_text,
    get_llm,
    invoke_llm_chat,
    normalize_reasoning_effort,
)
from app.core.config import settings
from app.core.prompts import (
    GROUNDED_ANSWER_DEVELOPER_PROMPT,
    build_grounded_answer_prompt,
)
from app.core.token_budget import ResponsesInputBudgeter
from app.indexing.vector_store import vector_store_manager
from app.retrieval.citation_builder import build_citations
from app.retrieval.conflict_detector import detect_conflicts
from app.retrieval.query_expansion import build_query_variants, should_expand_query
from app.retrieval.reranker import rerank_nodes

logger = logging.getLogger(__name__)

DEFAULT_EVIDENCE_LIMIT = 7
COMPARATIVE_EVIDENCE_LIMIT = 10
CONFLICT_EVIDENCE_LIMIT = 8
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
        conversation_context: dict[str, Any],
    ):
        self.client_id = client_id
        self.reasoning_effort = reasoning_effort
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
            response = invoke_llm_chat(
                model=settings.LLM_MODEL,
                input_messages=input_messages,
                reasoning_effort=self.reasoning_effort,
                max_output_tokens=settings.RESPONSE_MAX_OUTPUT_TOKENS,
                prompt_cache_key=settings.RESPONSE_PROMPT_CACHE_KEY,
                prompt_cache_retention=settings.RESPONSE_PROMPT_CACHE_RETENTION,
                safety_identifier=f"{settings.RESPONSE_SAFETY_IDENTIFIER_PREFIX}:{self.client_id}",
                user_tag=settings.RESPONSE_USER_TAG,
            )
            text = extract_chat_response_text(response)
            answer, reasoning = _split_reasoning_from_text(text)
            if not answer:
                raise ValueError("LLM returned empty answer")
            return GroundedAnswerResult(
                answer=answer,
                reasoning=reasoning,
                images_used=image_paths,
                reasoning_effort_applied=effort_applied,
            )
        except Exception:
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
        llm = get_llm(reasoning_effort=self.reasoning_effort)
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
        conversation_context: dict[str, Any] | None = None,
    ):
        self.client_id = client_id
        self.top_k = top_k
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self.conversation_context = conversation_context or {}
        self.prefetch_top_k = top_k + 5
        self.evidence_limit = DEFAULT_EVIDENCE_LIMIT
        self.comparative_evidence_limit = COMPARATIVE_EVIDENCE_LIMIT
        self.conflict_evidence_limit = CONFLICT_EVIDENCE_LIMIT
        self.filters = MetadataFilters(
            filters=[ExactMatchFilter(key="client_id", value=self.client_id)]
        )
        self.answer_synthesizer = GroundedAnswerSynthesizer(
            client_id=self.client_id,
            reasoning_effort=self.reasoning_effort,
            conversation_context=self.conversation_context,
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute the full retrieval and answer pipeline.

        Returns:
            Dict with answer, citations, conflicts, and metadata.
        """
        logger.info("Query for client %s: %s", self.client_id, question[:100])

        # Step 1: Hybrid Qdrant retrieval, with dense fallback for older indexes.
        source_nodes, retrieval_metadata = self._retrieve(question)

        if not source_nodes:
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
                **retrieval_metadata,
            }

        # Step 2: Rank candidates with semantic + temporal/version signals.
        ranked_nodes = self._rank_nodes(question, source_nodes)

        # Step 3: Select bounded evidence used for answer synthesis/citations.
        evidence_nodes = self._select_evidence_nodes(question, ranked_nodes)
        retrieval_diagnostics = _build_retrieval_diagnostics(
            ranked_nodes, evidence_nodes
        )

        # Step 4: Conflict detection (advisory, scoped to evidence + near-miss nodes).
        conflicts = detect_conflicts(
            ranked_nodes, evidence_nodes=evidence_nodes, question=question
        )

        # Step 5: Build citations only from the selected answer evidence.
        citations = build_citations(evidence_nodes)

        # Step 6: Synthesize grounded answer from selected evidence only.
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
            **retrieval_metadata,
        }

    def retrieve_only(self, question: str) -> List:
        """Retrieve source nodes without generating an answer."""
        source_nodes, _ = self._retrieve(question)
        return self._rank_nodes(question, source_nodes)

    def _rank_nodes(self, question: str, source_nodes: list) -> list:
        prefer_latest = not (
            _is_comparison_or_conflict_query(question)
            or _is_time_anchored_query(question)
        )
        rank_top_k = (
            self.top_k + 8 if _is_conflict_focused_query(question) else self.top_k
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
        return nodes, {"retrieval_mode": mode, "query_expanded": expanded}

    def _retrieve_with_mode(self, question: str, hybrid: bool) -> tuple[list, bool]:
        prefetch_top_k = (
            self.prefetch_top_k + 10
            if _is_conflict_focused_query(question)
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
        if should_expand:
            return self._retrieve_with_expansion(
                question,
                expansion_question,
                base_retriever,
            )

        return base_retriever.retrieve(question), False

    def _retrieve_with_expansion(
        self,
        question: str,
        expansion_question: dict[str, Any],
        base_retriever,
    ) -> tuple[list, bool]:
        query_variants = build_query_variants(expansion_question)
        if len(query_variants) == 1:
            return base_retriever.retrieve(question), False

        batches = [base_retriever.retrieve(variant) for variant in query_variants]
        return _fuse_node_batches(batches), True

    def _build_query_expansion_input(self, question: str) -> dict[str, Any]:
        return {
            "current_question": question,
            "recent_turns": self.conversation_context.get("recent_turns") or [],
        }

    def _select_evidence_nodes(self, question: str, ranked_nodes: list) -> list:
        if not ranked_nodes:
            return []

        comparative_query = _is_comparison_or_conflict_query(question)
        conflict_focused_query = _is_conflict_focused_query(question)
        evidence_cap = self._resolve_evidence_cap(
            comparative_query=comparative_query,
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
            comparative_query=comparative_query,
        )
        selected = _ensure_structured_evidence(
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
    return _safe_bool(raw, False)


def _node_has_image_assets(node: Any) -> bool:
    metadata = node.node.metadata or {}
    refs = metadata.get("asset_refs")
    if isinstance(refs, str):
        return bool(refs.strip())
    if isinstance(refs, list):
        return any(isinstance(ref, str) and ref.strip() for ref in refs)
    return False


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


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
    } or _safe_bool(metadata.get("table_detected"), False)


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
        or _safe_bool(metadata.get("chart_detected"), False)
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


def _build_retrieval_diagnostics(
    ranked_nodes: list[Any],
    evidence_nodes: list[Any],
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
        image_assets = citation.get("asset_refs") or []
        excerpt = _prompt_excerpt(citation)
        evidence_lines.append(
            f"[{index}] {label} | version={version} | chunk_type={chunk_type}"
            f"{location} | image_assets={len(image_assets)}\n"
            f"Excerpt:\n{excerpt}"
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
    )


def _build_labeled_context_sections(
    *,
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
        "conflict_lines": _build_context_conflict_lines(conflicts),
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
        excerpt = _prompt_excerpt(citation)
        lines.append(
            f"[{index}] {label} | version={version} | chunk_type={chunk_type}{location}\n"
            f"Excerpt: {excerpt}"
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
) -> list[dict[str, Any]]:
    if not image_paths:
        return input_messages

    messages = [*input_messages]
    if not messages:
        return messages

    user_message = messages[-1]
    if user_message.get("role") != "user":
        return messages

    user_text = str(user_message.get("content") or "")
    multimodal_content: list[dict[str, Any]] = [
        {"type": "input_text", "text": user_text}
    ]
    for path in image_paths:
        data_url = _image_path_to_data_url(path)
        if not data_url:
            continue
        multimodal_content.append({"type": "input_image", "image_url": data_url})

    user_message["content"] = multimodal_content
    return messages


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
    reasoning_parts: list[str] = []
    for block in getattr(message, "blocks", None) or []:
        if isinstance(block, ThinkingBlock):
            content = (block.content or "").strip()
            if content:
                reasoning_parts.append(content)
            continue
        if isinstance(block, TextBlock):
            content = (block.text or "").strip()
            if content:
                answer_parts.append(content)

    answer_text = "\n".join(answer_parts).strip()
    reasoning_text = "\n\n".join(reasoning_parts).strip() or None
    parsed_answer, parsed_reasoning = _split_reasoning_from_text(
        answer_text or str(getattr(message, "content", "") or "").strip()
    )
    if not reasoning_text and parsed_reasoning:
        reasoning_text = parsed_reasoning

    return parsed_answer, reasoning_text


def _split_reasoning_from_text(text: str) -> tuple[str, str | None]:
    if not text:
        return "", None

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
    else:
        answer = text.strip()

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
