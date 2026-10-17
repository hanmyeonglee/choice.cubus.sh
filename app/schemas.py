"""Request and response schemas with input serialization helpers."""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Name = Annotated[str, Field(min_length=1, max_length=64)]
Description = Annotated[str, Field(max_length=160)]


def state_text(state):
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def choice_labels(criteria):
    return [f"{name}: {description}" if description else name for name, description in criteria.items()]


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: str = Field(min_length=1, max_length=320, description="Question or statement to evaluate")


class ChoiceQuestion(Question):
    type: Literal["choice"]
    criteria: dict[Name, Description | None] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def unique_labels(self):
        labels = choice_labels(self.criteria)
        if len(set(labels)) != len(labels):
            raise ValueError("Combined choice names and descriptions must be unique.")
        return self


class NoulQuestion(Question):
    type: Literal["noul"]


QuestionType = Annotated[ChoiceQuestion | NoulQuestion, Field(discriminator="type")]


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "state": "I was charged twice for the same order.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {"billing": "Charges and invoices", "technical": "Technical issues"},
            },
            "duplicate_charge": {
                "type": "noul",
                "instructions": "The customer was charged twice for the same order.",
            },
        },
    }})
    state: str | dict[str, JsonValue] | list[JsonValue] = Field(
        description=(
            "Source material and context to evaluate. Use a string containing the original text "
            "or labeled sections by default. Objects and arrays are also accepted and serialized "
            "as JSON text. No specific keys or schema are required. Strings are passed unchanged. "
            "The serialized input must not exceed 2,000 characters."
        ),
        examples=[
            "I was charged twice for the same order.",
            {"message": "I was charged twice for the same order."},
            [{"speaker": "customer", "message": "I was charged twice for the same order."}],
        ],
    )
    questions: dict[Name, QuestionType] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def bounded_state(self):
        text = state_text(self.state)
        if not text.strip():
            raise ValueError("state must not be blank.")
        if len(text) > 2000:
            raise ValueError("state must not exceed 2,000 characters after serialization.")
        return self


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class EvaluateResponse(BaseModel):
    answers: dict[str, Annotated[ChoiceAnswer | NoulAnswer, Field(discriminator="type")]]
