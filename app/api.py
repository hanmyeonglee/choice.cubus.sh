"""A minimal single-choice classification API backed by GLiNER2.5-Decide."""

import logging
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

from .model import InputTooLongError, classify_text, load_model, resolve_dtype
from .schemas import ClassifyRequest, ClassifyResponse, render_text


def create_app():
    configured_keys = os.environ.get("API_KEYS", os.environ.get("API_KEY", ""))
    api_keys = tuple(key.strip().encode() for key in configured_keys.split(",") if key.strip())
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        if not api_keys or any(key.startswith(b"replace-with-") for key in api_keys):
            raise RuntimeError("Set API_KEYS to one or more real Bearer keys before starting the server.")
        device = os.environ.get("DEVICE", "cuda")
        precision = os.environ.get("PRECISION")
        dtype = resolve_dtype(device, precision)
        app.state.model = load_model(
            os.environ.get("MODEL_PATH", "/opt/models/gliner2.5-decide"),
            device,
            precision,
        )
        logging.info("Model ready: device=%s dtype=%s torch=%s", device, dtype, torch.__version__)
        yield
        del app.state.model

    app = FastAPI(
        title="Choice API",
        version="0.2.0",
        description=(
            "Classify text against candidate labels using GLiNER2.5-Decide (`fastino/GLiNER2.5-Decide`). "
            "The model is fixed on the server.\n\n"
            "### Writing requests\n\n"
            "Pass original prose or labeled sections in `text` (objects and arrays are also accepted and "
            "serialized as compact JSON). Supply 2–16 competing candidate labels in `labels` as either a "
            "list of strings or an object mapping each label to a short description. Optionally set `task` "
            "(defaults to `decision`) and `prompt` (a question or classification instruction).\n\n"
            "- For binary or yes/no questions, supply two meaningful competing labels (such as "
            '`["no", "yes"]` or `{"safe": "Read-only inspection", "unsafe": "Modifies or deletes state"}`).\n'
            "- Include concise label descriptions when category boundaries are domain-specific.\n"
            "- Serialized `text` is limited to 2,000 characters, and the complete prepared sequence "
            "(text, task, prompt, and labels) is limited to 1,024 subword tokens without silent truncation.\n\n"
            "### Results\n\n"
            "`choice` is the highest-scoring candidate label, and `probabilities` contains the native "
            "softmax score distribution across all candidate labels in request order. "
            "Scores are not calibrated probabilities of correctness."
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
        "/api/classify",
        response_model=ClassifyResponse,
        dependencies=[Depends(authenticate)],
        tags=["Classification"],
        summary="Classify text against competing candidate labels",
        responses={
            401: {"description": "Missing or invalid Bearer key"},
            422: {"description": "Invalid input or prepared sequence exceeds 1,024 tokens"},
        },
    )
    def classify(body: ClassifyRequest):
        text = render_text(body.text)
        with lock, torch.inference_mode():
            try:
                choice, probabilities = classify_text(
                    app.state.model,
                    text=text,
                    labels=body.labels,
                    task=body.task,
                    prompt=body.prompt,
                )
            except InputTooLongError as exc:
                raise HTTPException(422, str(exc)) from exc
        return ClassifyResponse(choice=choice, probabilities=probabilities)

    return app
