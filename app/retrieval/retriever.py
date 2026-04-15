"""
Retriever: orchestrates the full retrieval pipeline.

Flow: vector retrieval -> ranking -> evidence selection -> conflict checks -> citation building
"""

import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List

from llama_index.core import Settings
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
    ThinkingBlock,
)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from app.core.prompts import build_grounded_answer_prompt
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


class VecteraRetriever:
    """
    Full retrieval pipeline with client-scoped filtering,
    reranking, temporal awareness, conflict detection, and citations.
    """

    def __init__(self, client_id: str, top_k: int = 15):
        self.client_id = client_id
        self.top_k = top_k
        self.prefetch_top_k = top_k + 5
        self.evidence_limit = DEFAULT_EVIDENCE_LIMIT
        self.comparative_evidence_limit = COMPARATIVE_EVIDENCE_LIMIT
        self.conflict_evidence_limit = CONFLICT_EVIDENCE_LIMIT
        self.filters = MetadataFilters(
            filters=[ExactMatchFilter(key="client_id", value=self.client_id)]
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
            "retrieval_diagnostics": retrieval_diagnostics,
            **retrieval_metadata,
        }

    def retrieve_only(self, question: str) -> List:
        """Retrieve source nodes without generating an answer."""
        source_nodes, _ = self._retrieve(question)
        return self._rank_nodes(question, source_nodes)

    def _rank_nodes(self, question: str, source_nodes: list) -> list:
        prefer_latest = not _is_comparison_or_conflict_query(question)
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

        should_expand = should_expand_query(question)
        if should_expand:
            return self._retrieve_with_expansion(
                question, base_retriever, prefetch_top_k
            )

        return base_retriever.retrieve(question), False

    def _retrieve_with_expansion(
        self, question: str, base_retriever, prefetch_top_k: int
    ) -> tuple[list, bool]:
        try:
            fusion_retriever = QueryFusionRetriever(
                [base_retriever],
                similarity_top_k=prefetch_top_k,
                num_queries=3,
                mode=FUSION_MODES.RECIPROCAL_RANK,
                use_async=False,
                verbose=False,
            )
            return fusion_retriever.retrieve(question), True
        except Exception:
            logger.exception(
                "LlamaIndex query fusion expansion failed; using manual expansion fallback"
            )
            query_variants = build_query_variants(question)
            batches = [base_retriever.retrieve(variant) for variant in query_variants]
            return _fuse_node_batches(batches), len(query_variants) > 1

    def _select_evidence_nodes(self, question: str, ranked_nodes: list) -> list:
        if not ranked_nodes:
            return []

        comparative_query = _is_comparison_or_conflict_query(question)
        conflict_focused_query = _is_conflict_focused_query(question)
        evidence_cap = self.evidence_limit
        if comparative_query:
            evidence_cap = self.comparative_evidence_limit
        if conflict_focused_query:
            evidence_cap = self.conflict_evidence_limit

        candidate_nodes = ranked_nodes
        if conflict_focused_query:
            # For conflict queries, prioritize chunks that actually carry numeric signals.
            numeric_nodes = [node for node in ranked_nodes if _has_numeric_signal(node)]
            non_numeric_nodes = [
                node for node in ranked_nodes if not _has_numeric_signal(node)
            ]
            candidate_nodes = numeric_nodes + non_numeric_nodes

        primary_candidates = candidate_nodes
        secondary_candidates: list[Any] = []
        if not _is_reasoning_priority_query(question):
            primary_candidates = [
                node for node in candidate_nodes if not _is_reasoning_chunk(node)
            ]
            secondary_candidates = [
                node for node in candidate_nodes if _is_reasoning_chunk(node)
            ]

        selected = []
        selected_keys: set[str] = set()
        selected_diversity_keys: set[str] = set()
        candidate_batches = [primary_candidates]
        if secondary_candidates:
            candidate_batches.append(secondary_candidates)

        if comparative_query:
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
                        return selected

        for batch in candidate_batches:
            for node in batch:
                if len(selected) >= evidence_cap:
                    break
                key = _node_unique_key(node)
                if key in selected_keys:
                    continue
                selected.append(node)
                selected_keys.add(key)
            if len(selected) >= evidence_cap:
                break

        return selected

    def _synthesize_answer(
        self,
        question: str,
        citations: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not citations:
            return {
                "answer": (
                    "I could not find enough relevant evidence in the uploaded "
                    "documents to answer this question confidently."
                ),
                "reasoning": None,
                "images_used": [],
            }

        image_paths = _collect_image_evidence_paths(citations)
        prompt = _build_grounded_prompt(
            question,
            citations,
            conflicts,
            image_attachment_count=len(image_paths),
        )

        attempts = [
            ("multimodal", _build_grounded_message(prompt, image_paths), image_paths),
            ("text_only", _build_grounded_message(prompt, []), []),
        ]
        if not image_paths:
            attempts = attempts[1:]

        for attempt_name, message, used_images in attempts:
            try:
                response = self._chat_with_optional_thinking([message])
                answer, reasoning = _extract_answer_and_reasoning_from_chat(response)
                if answer:
                    return {
                        "answer": answer,
                        "reasoning": reasoning,
                        "images_used": used_images,
                    }
                raise ValueError("LLM returned empty answer")
            except Exception:
                logger.exception("%s answer synthesis failed", attempt_name)

        try:
            response = Settings.llm.complete(prompt)
            answer, reasoning = _split_reasoning_from_text(str(response).strip())
            if answer:
                return {
                    "answer": answer,
                    "reasoning": reasoning,
                    "images_used": [],
                }
            raise ValueError("LLM returned empty answer")
        except Exception:
            logger.exception(
                "Answer synthesis failed; returning source-grounded fallback"
            )
            return {
                "answer": _build_source_grounded_fallback(citations),
                "reasoning": None,
                "images_used": image_paths,
            }

    def _chat_with_optional_thinking(self, messages: list[ChatMessage]):
        return Settings.llm.chat(messages)


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
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "1", "yes"}
    return False


def _is_reasoning_chunk(node: Any) -> bool:
    metadata = node.node.metadata or {}
    return str(metadata.get("chunk_type") or "") in REASONING_CHUNK_TYPES


def _is_reasoning_priority_query(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in REASONING_PRIORITY_TERMS)


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

    return {
        "ranked_document_count": len(_document_diversity_keys(ranked_nodes)),
        "evidence_document_count": len(_document_diversity_keys(evidence_nodes)),
        "ranked_chunk_types": ranked_chunk_types,
        "evidence_chunk_types": evidence_chunk_types,
        "ranked_reasoning_count": ranked_reasoning,
        "evidence_reasoning_count": evidence_reasoning,
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
) -> str:
    evidence_lines = []
    for index, citation in enumerate(citations, start=1):
        label = citation.get("citation_label") or citation.get(
            "document_name", f"Source {index}"
        )
        version = citation.get("version_label") or "unknown"
        chunk_type = citation.get("chunk_type") or "text"
        location = ""
        if citation.get("page_num"):
            location = f" | page={citation['page_num']}"
        elif citation.get("slide_num"):
            location = f" | slide={citation['slide_num']}"
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

    return build_grounded_answer_prompt(
        question=question,
        image_attachment_count=image_attachment_count,
        evidence_block=evidence_block,
        conflict_block=conflict_block,
    )


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


def _collect_image_evidence_paths(
    citations: list[dict[str, Any]], max_images: int = MAX_MULTIMODAL_IMAGES
) -> list[str]:
    seen: set[str] = set()
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
            if path_str in seen:
                continue
            seen.add(path_str)
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
        elif isinstance(block, TextBlock):
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
