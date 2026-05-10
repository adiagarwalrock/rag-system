"""
Prompt construction, context formatting, and image path resolution.

Extracted from retriever.py to keep the prompt/formatting domain
separate from retrieval orchestration.
"""

import base64
import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import Any

from app.prompts.templates import build_grounded_answer_prompt
from app.services.evidence_selector import REASONING_CHUNK_TYPES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MULTIMODAL_IMAGES = 6
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
RECENT_TURN_MAX_CHARS = 320
CROSS_SESSION_USER_MAX_CHARS = 220
CROSS_SESSION_ASSISTANT_MAX_CHARS = 260


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_grounded_prompt(
    question: str,
    citations: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    image_attachment_count: int = 0,
    conversation_context: dict[str, Any] | None = None,
    citation_image_map: dict[int, list[int]] | None = None,
) -> str:
    evidence_lines = []
    for index, citation in enumerate(citations, start=1):
        label = _citation_label(citation, index)
        version = citation.get("version_label") or "unknown"
        chunk_type = citation.get("chunk_type") or "text"
        location = _citation_location(citation)
        excerpt = _prompt_excerpt(citation)
        image_tag = _build_image_tag(index, citation, citation_image_map)
        evidence_lines.append(
            f"[{index}] {label} | version={version} | chunk_type={chunk_type}"
            f"{location} | {image_tag}\n"
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
    _, citation_image_map = _collect_image_evidence_paths(citations)

    return {
        "session_summary": summary,
        "cross_session_lines": _build_context_cross_session_lines(cross_session_pairs),
        "evidence_lines": _build_context_evidence_lines(citations, citation_image_map),
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


# ---------------------------------------------------------------------------
# Context line builders
# ---------------------------------------------------------------------------


def _build_context_evidence_lines(
    citations: list[dict[str, Any]],
    citation_image_map: dict[int, list[int]] | None = None,
) -> list[str]:
    lines: list[str] = []
    for index, citation in enumerate(citations, start=1):
        label = _citation_label(citation, index)
        version = citation.get("version_label") or "unknown"
        chunk_type = citation.get("chunk_type") or "text"
        location = _citation_location(citation)
        excerpt = _prompt_excerpt(citation)
        image_tag = _build_image_tag(index, citation, citation_image_map)
        lines.append(
            f"[{index}] {label} | version={version} | chunk_type={chunk_type}{location} | {image_tag}\n"
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


# ---------------------------------------------------------------------------
# Text formatting helpers
# ---------------------------------------------------------------------------


def _normalize_inline_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _build_image_tag(
    citation_index: int,
    citation: dict[str, Any],
    citation_image_map: dict[int, list[int]] | None,
) -> str:
    if citation_image_map is not None:
        image_indices = citation_image_map.get(citation_index, [])
        if image_indices:
            image_scope = str(citation.get("image_scope") or "unknown")
            indices_str = ",".join(str(i) for i in image_indices)
            return f"attached_image_indices={indices_str} | image_scope={image_scope}"
        return "no_attached_image"
    # Fallback for callers that don't pass a map: show count from asset_refs
    refs = citation.get("asset_refs") or []
    return f"image_assets={len(refs) if isinstance(refs, list) else (1 if refs else 0)}"


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


# ---------------------------------------------------------------------------
# Image path resolution and encoding
# ---------------------------------------------------------------------------


def _collect_image_evidence_paths(
    citations: list[dict[str, Any]], max_images: int = MAX_MULTIMODAL_IMAGES
) -> tuple[list[str], dict[int, list[int]]]:
    """Collect image paths from citations, prioritizing figure crops over page screenshots.

    Returns:
        image_paths: ordered list of resolved image file paths
        citation_image_map: {citation_1based_index: [image_1based_indices]}
    """
    # Build a candidate list: (citation_1based_idx, ref, artifact_bundle_path, is_crop)
    # Crops come first so they win budget slots over full-page screenshots.
    crop_candidates: list[tuple[int, str, Any]] = []
    screenshot_candidates: list[tuple[int, str, Any]] = []

    for citation_idx, citation in enumerate(citations, start=1):
        refs = citation.get("asset_refs") or []
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list):
            continue
        artifact_bundle_path = citation.get("artifact_bundle_path")
        image_scope = str(citation.get("image_scope") or "")
        for ref in refs:
            if image_scope == "figure_crop" or _is_likely_crop_path(ref):
                crop_candidates.append((citation_idx, ref, artifact_bundle_path))
            else:
                screenshot_candidates.append((citation_idx, ref, artifact_bundle_path))

    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    image_paths: list[str] = []
    citation_image_map: dict[int, list[int]] = {}

    def _try_add(citation_idx: int, ref: str, artifact_bundle_path: Any) -> None:
        if len(image_paths) >= max_images:
            return
        resolved = _resolve_asset_path(ref, artifact_bundle_path)
        if resolved is None:
            return
        if resolved.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            return
        path_str = str(resolved)
        if path_str in seen_paths:
            # Already added — record the existing index for this citation too
            existing_idx = image_paths.index(path_str) + 1
            citation_image_map.setdefault(citation_idx, [])
            if existing_idx not in citation_image_map[citation_idx]:
                citation_image_map[citation_idx].append(existing_idx)
            return
        try:
            content_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
        except OSError:
            return
        if content_hash in seen_hashes:
            return
        seen_paths.add(path_str)
        seen_hashes.add(content_hash)
        image_paths.append(path_str)
        image_1based = len(image_paths)
        citation_image_map.setdefault(citation_idx, []).append(image_1based)

    for citation_idx, ref, bundle in crop_candidates:
        _try_add(citation_idx, ref, bundle)
    for citation_idx, ref, bundle in screenshot_candidates:
        _try_add(citation_idx, ref, bundle)

    return image_paths, citation_image_map


def _is_likely_crop_path(ref: str) -> bool:
    name = Path(ref).name.lower()
    return any(token in name for token in ("figure", "crop", "fig_", "chart_"))


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
