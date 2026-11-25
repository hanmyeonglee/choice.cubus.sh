"""A minimal single-choice classification API backed by GLiNER2.5-Decide."""

import logging
import os
import secrets
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .runtime import APP_VERSION, ChoiceRuntime
from .schemas import ClassifyRequest, ClassifyResponse, render_text
from .worker import ModelWorker, WorkerInputTooLongError


def _parse_api_keys(environ: Mapping[str, str] | None = None) -> tuple[bytes, ...]:
    env = os.environ if environ is None else environ
    configured_keys = env.get("API_KEYS", env.get("API_KEY", ""))
    return tuple(key.strip().encode() for key in configured_keys.split(",") if key.strip())


def validate_api_keys(api_keys: tuple[bytes, ...]) -> tuple[bytes, ...]:
    """Ensure at least one non-placeholder Bearer API key is configured."""
    if not api_keys or any(key.startswith(b"replace-with-") for key in api_keys):
        raise RuntimeError("Set API_KEYS to one or more real Bearer keys before starting the server.")
    return api_keys


def load_api_keys(environ: Mapping[str, str] | None = None) -> tuple[bytes, ...]:
    """Read and validate non-placeholder Bearer API keys from environment variables."""
    return validate_api_keys(_parse_api_keys(environ))


def create_app(runtime: ChoiceRuntime | None = None):
    active_runtime = ChoiceRuntime.from_env() if runtime is None else runtime
    api_keys = _parse_api_keys()

    @asynccontextmanager
    async def lifespan(app):
        validate_api_keys(api_keys)
        device = os.environ.get("DEVICE", "cuda")
        precision = os.environ.get("PRECISION")
        idle_timeout = active_runtime.initial_worker_idle_timeout()
        worker = ModelWorker(
            os.environ.get("MODEL_PATH", "/opt/models/gliner2.5-decide"),
            device=device,
            precision=precision,
            idle_timeout=idle_timeout,
        )
        active_runtime.attach_worker(worker)
        if worker.idle_timeout == 0:
            worker.start()
        else:
            logging.info(
                "Model worker configured in on-demand idle mode: device=%s precision=%s idle_timeout=%.0fs",
                device,
                precision or "auto",
                worker.idle_timeout,
            )
        app.state.worker = worker
        app.state.runtime = active_runtime
        try:
            yield
        finally:
            active_runtime.detach_worker()
            worker.close()

    app = FastAPI(
        title="Choice API",
        version=APP_VERSION,
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
    app.state.runtime = active_runtime
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

    def require_accepting_requests():
        if not active_runtime.snapshot().accept_requests:
            raise HTTPException(503, "Service is temporarily paused for maintenance.")

    @app.get("/", include_in_schema=False)
    async def homepage():
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    @app.get("/api/healthz", tags=["Server"], summary="Server readiness and model residency")
    async def healthz():
        return {"ok": True, "model_loaded": app.state.worker.is_loaded()}

    @app.post(
        "/api/classify",
        response_model=ClassifyResponse,
        dependencies=[Depends(authenticate), Depends(require_accepting_requests)],
        tags=["Classification"],
        summary="Classify text against competing candidate labels",
        responses={
            401: {"description": "Missing or invalid Bearer key"},
            422: {"description": "Invalid input or prepared sequence exceeds 1,024 tokens"},
            503: {"description": "Service is temporarily paused for maintenance"},
        },
    )
    def classify(body: ClassifyRequest):
        settings = active_runtime.snapshot()
        if not settings.accept_requests:
            raise HTTPException(503, "Service is temporarily paused for maintenance.")
        if len(body.labels) > settings.max_labels:
            raise HTTPException(
                422,
                f"labels must contain at most {settings.max_labels} candidate labels.",
            )
        text = render_text(body.text)
        if len(text) > settings.max_text_chars:
            raise HTTPException(
                422,
                f"text must not exceed {settings.max_text_chars:,} characters after serialization.",
            )
        try:
            choice, probabilities = app.state.worker.classify(
                text=text,
                labels=body.labels,
                task=body.task,
                prompt=body.prompt,
            )
        except WorkerInputTooLongError as exc:
            raise HTTPException(422, str(exc)) from exc
        return ClassifyResponse(choice=choice, probabilities=probabilities)

    return app
