"""Request and response schemas for the Amber 150M classification API."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, model_validator

from .runtime import MODEL_ID

MAX_BATCH_INPUTS = 64
MAX_TEXT_CHARS_CAP = 16000

TextPayload = str | dict[str, JsonValue] | list[JsonValue]
LabelName = Annotated[str, Field(min_length=1, max_length=64)]
Description = Annotated[str, Field(min_length=1, max_length=160)]
TaskName = Annotated[str, Field(min_length=1, max_length=64)]
PromptText = Annotated[str, Field(min_length=1, max_length=320)]


def render_text(value: TextPayload) -> str:
    """Serialize string, dict, or list payloads into compact JSON text without NaN/Infinity."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class ClassifyInputItem(BaseModel):
    """Structured input item matching the Jina AI classification input object shape."""

    model_config = ConfigDict(extra="forbid")

    text: TextPayload = Field(
        description=(
            "Source material and context to classify. Pass original prose or labeled sections by "
            "default; JSON objects and arrays are also accepted and serialized as compact JSON text."
        ),
    )


ClassifyInput = (
    str
    | ClassifyInputItem
    | Annotated[list[str | ClassifyInputItem], Field(min_length=1, max_length=MAX_BATCH_INPUTS)]
)


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "model": MODEL_ID,
                "input": [
                    "I was charged twice for the same order.",
                    {"text": "The checkout page returns a 502 error."},
                ],
                "task": "department",
                "prompt": "Which team should handle this request?",
                "labels": {
                    "billing": "Charges, refunds, and invoices",
                    "technical": "Website, login, or technical outages",
                },
            }
        },
    )

    model: str = Field(
        default=MODEL_ID,
        description=(
            'Model identifier. Defaults to "amber-150m" (also accepts "MLukauskas/amber-150m"). '
            "Unknown model names are rejected with HTTP 400."
        ),
        examples=[MODEL_ID, "MLukauskas/amber-150m"],
    )
    input: ClassifyInput = Field(
        description=(
            "Input text or batch of inputs to classify. Accepts a single string, a single "
            '{"text": str | dict | list} object, or a non-empty list of 1–64 strings and/or '
            '{"text": ...} objects. Each serialized item must not exceed 16,000 characters '
            "(subject to the active runtime max_text_chars setting), and each complete prepared "
            "sequence (input + task + prompt + labels) must not exceed 7,999 subword tokens."
        ),
        examples=[
            "I was charged twice for the same order.",
            {"text": {"command": "git status -s", "cwd": "/workspace/app"}},
            ["I was charged twice.", {"text": "Login returns 500."}],
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
    _rendered_inputs: list[str] | None = PrivateAttr(default=None)

    def _compute_rendered_inputs(self) -> list[str]:
        raw_items = self.input if isinstance(self.input, list) else [self.input]
        rendered: list[str] = []
        for idx, item in enumerate(raw_items):
            payload = item.text if isinstance(item, ClassifyInputItem) else item
            text = render_text(payload)
            if not text.strip():
                raise ValueError(f"input[{idx}] must not be blank.")
            if len(text) > MAX_TEXT_CHARS_CAP:
                raise ValueError(
                    f"input[{idx}] must not exceed {MAX_TEXT_CHARS_CAP:,} characters after serialization."
                )
            rendered.append(text)
        return rendered

    @property
    def rendered_inputs(self) -> list[str]:
        """Return the cached serialized input strings produced during request validation."""
        if self._rendered_inputs is None:
            self._rendered_inputs = self._compute_rendered_inputs()
        return self._rendered_inputs

    @model_validator(mode="after")
    def validate_inputs(self) -> ClassifyRequest:
        self._rendered_inputs = self._compute_rendered_inputs()
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


class PredictionScore(BaseModel):
    label: str = Field(description="Candidate label name.")
    score: float = Field(description="Native softmax score for this candidate label.")


class ClassificationResult(BaseModel):
    object: Literal["classification"] = "classification"
    index: int = Field(description="Zero-based index corresponding to the input item.")
    prediction: str = Field(description="Highest-scoring candidate label.")
    score: float = Field(description="Softmax score of the winning prediction.")
    predictions: list[PredictionScore] = Field(
        description=(
            "All candidate labels sorted descending by score (ties preserve requested label order). "
            "Scores sum to 1 (up to floating-point precision) and are not calibrated probabilities of truth."
        )
    )


class UsageInfo(BaseModel):
    prompt_tokens: int = Field(description="Total prepared subword tokens processed across all input items.")
    total_tokens: int = Field(description="Total tokens processed (equal to prompt_tokens for classification).")


class ClassifyResponse(BaseModel):
    object: Literal["list"] = "list"
    model: str = Field(description="Canonical model identifier used for inference.")
    data: list[ClassificationResult] = Field(description="Ordered classification results for each input item.")
    usage: UsageInfo = Field(description="Token usage summary for the request.")


class ModelCard(BaseModel):
    id: str = Field(description="Model identifier.")
    object: Literal["model"] = "model"
    owned_by: str = Field(description="Model publisher.")


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard] = Field(description="Available classification models.")


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


def unpack_classify_payload(payload: Any) -> tuple[str, dict[str, float], int]:
    """Unpack either a 3-tuple (choice, probs, token_count) or legacy 2-tuple (choice, probs)."""
    if isinstance(payload, tuple) and len(payload) == 3:
        choice, probabilities, token_count = payload
        if not isinstance(probabilities, Mapping):
            raise ValueError("Invalid probabilities mapping in classification payload.")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise ValueError("Invalid token count in classification payload.")
        return str(choice), dict(probabilities), token_count
    if isinstance(payload, tuple) and len(payload) == 2:
        choice, probabilities = payload
        if not isinstance(probabilities, Mapping):
            raise ValueError("Invalid probabilities mapping in classification payload.")
        return str(choice), dict(probabilities), 0
    raise ValueError("Unexpected classification payload shape.")


def build_classification_result(index: int, probabilities: Mapping[str, float]) -> ClassificationResult:
    """Build a ClassificationResult with predictions sorted descending (ties preserve request order)."""
    if not isinstance(probabilities, Mapping) or not probabilities:
        raise ValueError("Probabilities mapping must not be empty.")
    validated: list[tuple[str, float]] = []
    for label, score in probabilities.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Classification scores must be finite numbers.")
        validated.append((str(label), float(score)))
    ordered = sorted(validated, key=lambda kv: -kv[1])
    top_label, top_score = ordered[0]
    return ClassificationResult(
        object="classification",
        index=index,
        prediction=top_label,
        score=top_score,
        predictions=[PredictionScore(label=label, score=score) for label, score in ordered],
    )
