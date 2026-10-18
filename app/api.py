"""A small choice/noul API backed directly by GLiFormer."""

import logging
import math
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated

import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .model import load_model
from .schemas import (
    ChoiceAnswer,
    ChoiceQuestion,
    EvaluateRequest,
    EvaluateResponse,
    NoulAnswer,
    NoulQuestion,
    choice_labels,
    state_text,
)


def create_app():
    configured_keys = os.environ.get("API_KEYS", os.environ.get("API_KEY", ""))
    api_keys = tuple(key.strip().encode() for key in configured_keys.split(",") if key.strip())
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        if not api_keys or any(key.startswith(b"replace-with-") for key in api_keys):
            raise RuntimeError("Set API_KEYS to one or more real Bearer keys before starting the server.")
        device = os.environ.get("DEVICE", "cuda")
        app.state.model = load_model(
            os.environ.get("MODEL_PATH", "/opt/models/gliformer-base-v1"), device
        )
        logging.info("Model ready: device=%s dtype=float32 torch=%s", device, torch.__version__)
        yield
        del app.state.model

    app = FastAPI(
        title="Choice API",
        version="0.1.0",
        description=(
            "Evaluate choice and noul questions against state. The model is fixed on the server.\n\n"
            "### Writing state\n\n"
            "Use a string containing the original prose or labeled sections by default. "
            "Objects and arrays are also accepted; no specific keys or schema are required. "
            "Strings are passed unchanged, while objects and arrays are serialized as JSON text. "
            "The serialized input is limited to 2,000 characters and is never truncated automatically.\n\n"
            "- Pass original documents and reviews, or short labels and values for multiple attributes.\n"
            "- Identify speakers and message order in conversations, and times and current status in histories.\n"
            "- Separate claims from verified facts and preserve times, units, and negation.\n"
            "- Put source material in state, and questions and evaluation criteria in questions.\n\n"
            "JSON does not guarantee better accuracy. "
            "The model's published evaluation results focus on English.\n\n"
            "### Results\n\n"
            "Up to four questions are processed sequentially per request. "
            "choice scores are normalized by their sum. noul is the raw sigmoid score from "
            "evaluating the question statement as a single class. "
            "Neither value is calibrated as a probability of correctness."
        ),
        lifespan=lifespan,
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    )
    bearer = HTTPBearer(
        auto_error=False,
        scheme_name="BearerKey",
        description="Any key configured in API_KEYS (or the legacy API_KEY setting)",
    )

    def authenticate(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        valid_key = False
        if credentials is not None:
            provided_key = credentials.credentials.encode()
            # Compare every configured key without stopping after a match.
            for api_key in api_keys:
                valid_key |= secrets.compare_digest(provided_key, api_key)
        if not valid_key:
            raise HTTPException(401, "Invalid or missing Bearer key", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/", include_in_schema=False)
    async def homepage():
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    @app.get("/api/healthz", tags=["Server"], summary="Readiness after model loading")
    async def healthz():
        return {"ok": True}

    @app.post(
        "/api/evaluate",
        response_model=EvaluateResponse,
        dependencies=[Depends(authenticate)],
        tags=["Evaluation"],
        summary="Evaluate choice and noul questions against state",
        responses={401: {"description": "Missing or invalid Bearer key"}},
    )
    def evaluate(body: EvaluateRequest):
        text = state_text(body.state)
        answers = {}
        # Separate forwards keep questions independent and GPU usage sequential.
        with lock, torch.inference_mode():
            for question_id, question in body.questions.items():
                if isinstance(question, ChoiceQuestion):
                    labels = choice_labels(question.criteria)
                    classes = {question.instructions: labels}
                else:
                    labels = [question.instructions]
                    classes = labels
                predictions = app.state.model.classify(
                    text, classes, threshold=-1.0, multi_label=True, batch_size=1
                )
                scores = {item["class_name"]: item["score"] for item in predictions}
                if isinstance(question, NoulQuestion):
                    answers[question_id] = NoulAnswer(noul=scores[labels[0]])
                else:
                    total = sum(scores[label] for label in labels)
                    if not math.isfinite(total) or total <= 0:
                        raise HTTPException(503, "Model returned invalid choice scores")
                    probabilities = {
                        name: scores[label] / total
                        for name, label in zip(question.criteria, labels)
                    }
                    answers[question_id] = ChoiceAnswer(
                        choice=max(probabilities, key=probabilities.get),
                        probabilities=probabilities,
                    )
        return EvaluateResponse(answers=answers)

    return app
