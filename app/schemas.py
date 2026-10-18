"""Request and response schemas for the GLiNER2.5-Decide classification API."""

import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

LabelName = Annotated[str, Field(min_length=1, max_length=64)]
Description = Annotated[str, Field(min_length=1, max_length=160)]
TaskName = Annotated[str, Field(min_length=1, max_length=64)]
PromptText = Annotated[str, Field(min_length=1, max_length=320)]


def render_text(value: str | dict[str, JsonValue] | list[JsonValue]) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "text": "I was charged twice for the same order.",
                "task": "department",
                "prompt": "Which team should handle this request?",
                "labels": {
                    "billing": "Charges, refunds, and invoices",
                    "technical": "Website, login, or technical outages",
                },
            }
        },
    )

    text: str | dict[str, JsonValue] | list[JsonValue] = Field(
        description=(
            "Source material and context to classify. Use a string containing original prose "
            "or labeled sections by default. Objects and arrays are also accepted and serialized "
            "as compact JSON text. The serialized text must not exceed 2,000 characters, and the "
            "complete prepared sequence (text + task + prompt + labels) must not exceed 1,024 tokens."
        ),
        examples=[
            "I was charged twice for the same order.",
            {"command": "git status -s", "cwd": "/workspace/app"},
        ],
    )
    labels: list[LabelName] | dict[LabelName, Description | None] = Field(
        min_length=2,
        max_length=16,
        description=(
            "2 to 16 distinct candidate labels. Pass either a list of label names "
            '(e.g. ["billing", "technical"]) or an object mapping label names to optional '
            "descriptions. For binary yes/no decisions, pass two meaningful competing labels "
            '(e.g. ["no", "yes"] or {"safe": "Read-only command", "unsafe": "Modifies state"}).'
        ),
    )
    task: TaskName = Field(
        default="decision",
        description="Task or category group name passed to the classification schema.",
    )
    prompt: PromptText | None = Field(
        default=None,
        description="Optional question or classification instruction describing the decision.",
    )

    @model_validator(mode="after")
    def validate_inputs(self):
        rendered = render_text(self.text)
        if not rendered.strip():
            raise ValueError("text must not be blank.")
        if len(rendered) > 2000:
            raise ValueError("text must not exceed 2,000 characters after serialization.")
        if not self.task.strip():
            raise ValueError("task must not be blank.")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("prompt must not be blank when provided.")

        if isinstance(self.labels, list):
            if any(not name.strip() for name in self.labels):
                raise ValueError("Label names must not be blank.")
            if len(set(self.labels)) != len(self.labels):
                raise ValueError("Label names must be unique.")
        else:
            for name, desc in self.labels.items():
                if not name.strip():
                    raise ValueError("Label names must not be blank.")
                if desc is not None and not desc.strip():
                    raise ValueError("Label descriptions must not be blank when provided.")
        return self


class ClassifyResponse(BaseModel):
    choice: str = Field(description="Highest-scoring candidate label.")
    probabilities: dict[str, float] = Field(
        description=(
            "Native softmax scores across all candidate labels in their requested order. "
            "Scores sum to 1 (up to floating-point precision) and are not calibrated probabilities of truth."
        )
    )
