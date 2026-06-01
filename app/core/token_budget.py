from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import tiktoken

from app.core.config import settings
from app.core.prompts import build_budgeted_grounded_answer_prompt

KNOWN_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-5.2": 200000,
    "gpt-5.4": 200000,
    "gpt-5.4-mini": 128000,
}

MIN_INPUT_RATIO = 0.1
MAX_INPUT_RATIO = 0.95
MESSAGE_OVERHEAD_TOKENS = 8
PER_TURN_CONTENT_TOKEN_CAP = 240
MIN_REMAINING_BUDGET_TOKENS = 256
HISTORY_BUDGET_RATIO = 0.35
SUMMARY_BUDGET_RATIO = 0.10
CROSS_BUDGET_RATIO = 0.15
CONFLICT_BUDGET_RATIO = 0.10
MIN_EVIDENCE_BUDGET_TOKENS = 128
MIN_USER_CONTEXT_TOKENS = 64
DEFAULT_TOKEN_ENCODING = "o200k_base"


def model_context_window(model: str) -> int:
    normalized = (model or "").strip().lower()
    return KNOWN_CONTEXT_WINDOWS.get(normalized, settings.LLM_CONTEXT_WINDOW_TOKENS)


def get_token_encoding(
    *,
    encoding_name: str | None = None,
    model: str | None = None,
):
    preferred = (encoding_name or settings.TOKEN_BUDGET_ENCODING or "").strip()
    if preferred:
        try:
            return tiktoken.get_encoding(preferred)
        except Exception:
            pass

    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            pass

    try:
        return tiktoken.get_encoding(DEFAULT_TOKEN_ENCODING)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(
    text: str,
    *,
    encoding_name: str | None = None,
    model: str | None = None,
) -> int:
    if not text:
        return 0
    encoding = get_token_encoding(encoding_name=encoding_name, model=model)
    return len(encoding.encode(text))


def truncate_text_by_tokens(
    text: str,
    *,
    max_tokens: int,
    encoding_name: str | None = None,
    model: str | None = None,
) -> str:
    if max_tokens <= 0 or not text:
        return ""
    encoding = get_token_encoding(encoding_name=encoding_name, model=model)
    token_ids = encoding.encode(text)
    if len(token_ids) <= max_tokens:
        return text
    return encoding.decode(token_ids[:max_tokens]).rstrip()


@dataclass(frozen=True)
class BudgetMetrics:
    model: str
    context_window: int
    input_budget_tokens: int
    total_input_tokens: int
    history_tokens: int
    summary_tokens: int
    cross_session_tokens: int
    evidence_tokens: int
    conflict_tokens: int


class ResponsesInputBudgeter:
    """
    Build token-capped context payloads for Responses API input.

    We cap prompt input to a fixed fraction of model context length, then
    allocate per-section budgets and trim deterministically.
    """

    def __init__(
        self,
        *,
        model: str,
        ratio: float | None = None,
    ):
        self.model = model
        self.context_window = model_context_window(model)
        self.ratio = max(
            MIN_INPUT_RATIO,
            min(ratio or settings.RESPONSE_INPUT_BUDGET_RATIO, MAX_INPUT_RATIO),
        )
        self.encoding = get_token_encoding(model=model)

    @property
    def input_budget_tokens(self) -> int:
        return int(math.floor(self.context_window * self.ratio))

    def count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content")
            if isinstance(content, str):
                total += self.count_text_tokens(f"{role}\n{content}\n")
                continue
            if isinstance(content, list):
                flattened = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "input_text":
                        flattened.append(str(item.get("text") or ""))
                    elif item.get("type") == "input_image":
                        flattened.append("[IMAGE]")
                total += self.count_text_tokens(f"{role}\n" + "\n".join(flattened))
                continue
            total += self.count_text_tokens(f"{role}\n{str(content or '')}\n")
        # Add lightweight envelope overhead per message.
        return total + (len(messages) * MESSAGE_OVERHEAD_TOKENS)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        token_ids = self.encoding.encode(text)
        if len(token_ids) <= max_tokens:
            return text
        truncated = self.encoding.decode(token_ids[:max_tokens])
        return truncated.rstrip()

    def trim_recent_history(
        self,
        recent_turns: list[dict[str, str]],
        max_tokens: int,
    ) -> list[dict[str, str]]:
        if max_tokens <= 0:
            return []

        selected: list[dict[str, str]] = []
        running_tokens = 0

        for turn in reversed(recent_turns):
            role = str(turn.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = " ".join(str(turn.get("content") or "").split())
            if not content:
                continue
            content = self.truncate_to_tokens(content, PER_TURN_CONTENT_TOKEN_CAP)
            candidate = {"role": role, "content": content}
            delta_tokens = self.count_messages_tokens([candidate])
            if running_tokens + delta_tokens > max_tokens:
                break
            selected.append(candidate)
            running_tokens += delta_tokens

        return list(reversed(selected))

    def build_budgeted_sections(
        self,
        *,
        developer_prompt: str,
        question: str,
        recent_turns: list[dict[str, str]],
        session_summary: str,
        cross_session_lines: list[str],
        evidence_lines: list[str],
        answering_notes_lines: list[str],
        conflict_lines: list[str],
    ) -> tuple[list[dict[str, Any]], str, BudgetMetrics]:
        total_budget = self.input_budget_tokens

        # Reserve prompt envelope for static + final user query framing.
        base_messages = [
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": f"CURRENT_QUERY:\n{question}"},
        ]
        base_tokens = self.count_messages_tokens(base_messages)
        remaining = max(MIN_REMAINING_BUDGET_TOKENS, total_budget - base_tokens)

        history_budget = int(remaining * HISTORY_BUDGET_RATIO)
        summary_budget = int(remaining * SUMMARY_BUDGET_RATIO)
        cross_budget = int(remaining * CROSS_BUDGET_RATIO)
        conflict_budget = int(remaining * CONFLICT_BUDGET_RATIO)
        answering_notes_budget = int(remaining * CONFLICT_BUDGET_RATIO)
        evidence_budget = max(
            MIN_EVIDENCE_BUDGET_TOKENS,
            remaining
            - history_budget
            - summary_budget
            - cross_budget
            - conflict_budget
            - answering_notes_budget,
        )

        history_messages = self.trim_recent_history(recent_turns, history_budget)
        summary_text = self.truncate_to_tokens(session_summary, summary_budget)
        cross_text = self._fit_lines(cross_session_lines, cross_budget)
        conflict_text = self._fit_lines(conflict_lines, conflict_budget)
        answering_notes_text = self._fit_lines(answering_notes_lines, answering_notes_budget)
        evidence_text = self._fit_lines(evidence_lines, evidence_budget)

        user_context = build_budgeted_grounded_answer_prompt(
            question=question,
            session_summary=summary_text,
            cross_session_block=cross_text,
            answering_notes_block=answering_notes_text,
            evidence_block=evidence_text,
            conflict_block=conflict_text,
        )

        # Final hard trim to guarantee <= budget.
        final_messages: list[dict[str, Any]] = [
            {"role": "developer", "content": developer_prompt},
            *history_messages,
            {"role": "user", "content": user_context},
        ]
        total_tokens = self.count_messages_tokens(final_messages)
        if total_tokens > total_budget:
            # Keep developer + history; trim dynamic user context.
            fixed_tokens = self.count_messages_tokens(final_messages[:-1])
            allowed_user_tokens = max(
                MIN_USER_CONTEXT_TOKENS, total_budget - fixed_tokens
            )
            user_context = self.truncate_to_tokens(user_context, allowed_user_tokens)
            final_messages[-1]["content"] = user_context
            total_tokens = self.count_messages_tokens(final_messages)

        if total_tokens > total_budget:
            # Drop oldest history until we fit.
            while len(final_messages) > 2 and total_tokens > total_budget:
                del final_messages[1]
                total_tokens = self.count_messages_tokens(final_messages)

        metrics = BudgetMetrics(
            model=self.model,
            context_window=self.context_window,
            input_budget_tokens=total_budget,
            total_input_tokens=total_tokens,
            history_tokens=self.count_messages_tokens(history_messages),
            summary_tokens=self.count_text_tokens(summary_text),
            cross_session_tokens=self.count_text_tokens(cross_text),
            evidence_tokens=self.count_text_tokens(evidence_text),
            conflict_tokens=self.count_text_tokens(conflict_text),
        )
        return final_messages, user_context, metrics

    def _fit_lines(self, lines: list[str], max_tokens: int) -> str:
        if max_tokens <= 0 or not lines:
            return ""
        selected: list[str] = []
        running = 0
        for line in lines:
            candidate = line.strip()
            if not candidate:
                continue
            token_cost = self.count_text_tokens(candidate + "\n")
            if running + token_cost > max_tokens:
                break
            selected.append(candidate)
            running += token_cost
        return "\n".join(selected).strip()


def _encoding_for_model(model: str):
    return get_token_encoding(model=model)
