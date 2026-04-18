from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import tiktoken

from app.core.config import settings

KNOWN_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-5.2": 200000,
    "gpt-5.4": 200000,
    "gpt-5.4-mini": 128000,
}


def model_context_window(model: str) -> int:
    normalized = (model or "").strip().lower()
    return KNOWN_CONTEXT_WINDOWS.get(normalized, settings.LLM_CONTEXT_WINDOW_TOKENS)


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
        self.ratio = max(0.1, min(ratio or settings.RESPONSE_INPUT_BUDGET_RATIO, 0.95))
        self.encoding = _encoding_for_model(model)

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
        return total + (len(messages) * 8)

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
            content = self.truncate_to_tokens(content, 240)
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
        conflict_lines: list[str],
    ) -> tuple[list[dict[str, Any]], str, BudgetMetrics]:
        total_budget = self.input_budget_tokens

        # Reserve prompt envelope for static + final user query framing.
        base_messages = [
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": f"CURRENT_QUERY:\n{question}"},
        ]
        base_tokens = self.count_messages_tokens(base_messages)
        remaining = max(256, total_budget - base_tokens)

        history_budget = int(remaining * 0.35)
        summary_budget = int(remaining * 0.10)
        cross_budget = int(remaining * 0.15)
        conflict_budget = int(remaining * 0.10)
        evidence_budget = max(128, remaining - history_budget - summary_budget - cross_budget - conflict_budget)

        history_messages = self.trim_recent_history(recent_turns, history_budget)
        summary_text = self.truncate_to_tokens(session_summary, summary_budget)
        cross_text = self._fit_lines(cross_session_lines, cross_budget)
        conflict_text = self._fit_lines(conflict_lines, conflict_budget)
        evidence_text = self._fit_lines(evidence_lines, evidence_budget)

        user_context = "\n\n".join(
            [
                f"CURRENT_QUERY:\n{question}",
                f"SESSION_SUMMARY:\n{summary_text or '(none)'}",
                "CURRENT_SESSION_RECENT_TURNS:\n"
                + (
                    "\n".join(
                        f"- {msg['role'].upper()}: {msg['content']}" for msg in history_messages
                    )
                    if history_messages
                    else "- (none)"
                ),
                f"CROSS_SESSION_RELEVANT_QA:\n{cross_text or '- (none)'}",
                f"RETRIEVAL_EVIDENCE:\n{evidence_text or '- (none)'}",
                f"CONFLICT_HINTS:\n{conflict_text or '- (none)'}",
            ]
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
            allowed_user_tokens = max(64, total_budget - fixed_tokens)
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
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        # Use a modern fallback tokenizer for unknown model aliases.
        return tiktoken.get_encoding("o200k_base")
