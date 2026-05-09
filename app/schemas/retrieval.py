from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_TRUNCATION_SIGNALS = re.compile(r"(\.\.\.$|\[trunc|\(trunc|\bcontinued\.\.\.$)", re.IGNORECASE)
_COT_TAG_PATTERN = re.compile(r"</?\s*(thinking|answer)\s*>", re.IGNORECASE)
_COT_LEADING_PATTERN = re.compile(
    r"(?im)^\s*(?:let me (?:think|analy(?:ze|se)|break|consider)|"
    r"to answer this|thinking:|analysis:|step\s+\d+[:.)])"
)
_COT_REASONING_LIST_PATTERN = re.compile(r"(?im)^\s*(?:step\s+\d+[:.)]|\d+[\).]\s+)")
_JSON_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)


class GroundedAnswerStructuredResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    _MAX_REASONING_STEPS: ClassVar[int] = 5

    reasoning: list[str] = Field(
        default_factory=list,
        description=(
            "Concise evidence-based justification bullets; "
            "not private chain-of-thought."
        ),
    )
    answer: str = Field(
        min_length=1,
        max_length=4000,
        description="Final grounded answer with inline citations like [1].",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_reasoning_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        reasoning = normalized.get("reasoning")
        if isinstance(reasoning, str):
            normalized["reasoning"] = [reasoning]
        elif reasoning is None:
            normalized["reasoning"] = []
        return normalized

    @staticmethod
    def _strip_json_block(value: str) -> str:
        match: re.Match[str] | None = _JSON_BLOCK_PATTERN.match(value.strip())
        return match.group(1).strip() if match else value.strip()

    @classmethod
    def _clean_visible_text(
        cls,
        value: str,
        field_name: str,
        *,
        reject_truncation: bool = True,
    ) -> str:
        cleaned = cls._strip_json_block(value)
        if not cleaned:
            raise ValueError(f"{field_name} cannot be empty")
        if reject_truncation and _TRUNCATION_SIGNALS.search(cleaned):
            raise ValueError(f"{field_name} appears to be truncated")
        if _COT_TAG_PATTERN.search(cleaned):
            raise ValueError(f"{field_name} must not contain <thinking>/<answer> tags")
        if _COT_LEADING_PATTERN.search(cleaned):
            raise ValueError(f"{field_name} must not contain chain-of-thought lead-ins")
        return cleaned

    @field_validator("answer")
    @classmethod
    def _validate_answer(cls, value: str) -> str:
        return cls._clean_visible_text(value, "answer")

    @field_validator("reasoning")
    @classmethod
    def _validate_reasoning(cls, value: list[str]) -> list[str]:
        cleaned_steps: list[str] = []
        for raw_step in value[: cls._MAX_REASONING_STEPS]:
            try:
                step = cls._clean_visible_text(
                    raw_step,
                    "reasoning item",
                    reject_truncation=False,
                )
            except ValueError:
                continue

            if _COT_REASONING_LIST_PATTERN.search(step):
                continue
            if step:
                cleaned_steps.append(step)

        return cleaned_steps


@dataclass(frozen=True, slots=True)
class GroundedAnswerResult:
    answer: str
    reasoning: str = ""
    images_used: list[str] = field(default_factory=list)
    reasoning_effort_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "reasoning": self.reasoning,
            "images_used": self.images_used,
            "reasoning_effort_applied": self.reasoning_effort_applied,
        }


def format_reasoning_bullets(reasoning: list[str]) -> str:
    cleaned = [item.strip() for item in reasoning if isinstance(item, str) and item.strip()]
    return "\n".join(cleaned)
