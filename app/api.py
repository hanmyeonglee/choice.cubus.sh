"""A minimal single-choice classification API backed by Amber 150M."""

import inspect
import json
import logging
import os
import secrets
import threading
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .runtime import APP_VERSION, MODEL_ID, ChoiceRuntime
from .schemas import ClassifyRequest, ClassifyResponse
from .worker import ModelWorker, WorkerCancelledError, WorkerInputTooLongError


def _supports_kwarg(fn: Any, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


MAX_PUBLIC_BODY_BYTES = 128 << 10  # 128 KiB
PUBLIC_CLASSIFY_CONCURRENCY = 24


async def _send_public_json_error(send: Any, status_code: int, detail: str) -> None:
    body = json.dumps({"detail": detail}, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class _PublicClassifyLimitMiddleware:
    """Enforce a strict body-size cap on POST /api/classify before JSON parsing."""

    def __init__(self, app: Any, max_bytes: int = MAX_PUBLIC_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or str(scope.get("method", "")).upper() != "POST"
            or scope.get("path") != "/api/classify"
        ):
            await self.app(scope, receive, send)
            return

        cl_headers = [val for name, val in scope.get("headers", ()) if name.lower() == b"content-length"]
        if len(cl_headers) > 1:
            await _send_public_json_error(send, 400, "Invalid Content-Length header.")
            return
        if len(cl_headers) == 1:
            cl_str = cl_headers[0].decode("latin-1").strip()
            if not cl_str.isascii() or not cl_str.isdigit():
                await _send_public_json_error(send, 400, "Invalid Content-Length header.")
                return
            if int(cl_str) > self.max_bytes:
                await _send_public_json_error(send, 413, "Request body too large.")
                return

        buf = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            msg_type = message.get("type")
            if msg_type == "http.disconnect":
                async def disconnect_receive() -> dict[str, Any]:
                    return {"type": "http.disconnect"}

                await self.app(scope, disconnect_receive, send)
                return
            if msg_type == "http.request":
                chunk = message.get("body", b"")
                if len(buf) + len(chunk) > self.max_bytes:
                    await _send_public_json_error(send, 413, "Request body too large.")
                    return
                buf.extend(chunk)
                more_body = bool(message.get("more_body", False))

        body_sent = False
        payload = bytes(buf)

        async def replay_receive() -> dict[str, Any]:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": payload, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


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
            os.environ.get("MODEL_PATH", "/opt/models/amber-150m"),
            device=device,
            precision=precision,
            idle_timeout=idle_timeout,
        )
        active_runtime.attach_worker(worker)
        app.state.worker = worker
        app.state.runtime = active_runtime
        try:
            if worker.idle_timeout == 0:
                await anyio.to_thread.run_sync(worker.start, abandon_on_cancel=True)
            else:
                logging.info(
                    "Model worker configured in on-demand idle mode: device=%s precision=%s idle_timeout=%.0fs",
                    device,
                    precision or "auto",
                    worker.idle_timeout,
                )
            yield
        finally:
            active_runtime.detach_worker()
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(worker.close)

    app = FastAPI(
        title="Choice API",
        version=APP_VERSION,
        description=(
            "Classify text against candidate labels using Amber 150M (`MLukauskas/amber-150m`). "
            "The model is fixed on the server.\n\n"
            "### Writing requests\n\n"
            "Pass original prose or labeled sections in `text` (objects and arrays are also accepted and "
            "serialized as compact JSON). Supply 2–16 competing candidate labels in `labels` as either a "
            "list of strings or an object mapping each label to a short description. Optionally set `task` "
            "(defaults to `decision`) and `prompt` (a question or classification instruction).\n\n"
            "- For binary or yes/no questions, supply two meaningful competing labels (such as "
            '`["no", "yes"]` or `{"safe": "Read-only inspection", "unsafe": "Modifies or deletes state"}`).\n'
            "- Include concise label descriptions when category boundaries are domain-specific.\n"
            "- Serialized `text` defaults to an 8,000-character limit (configurable up to 16,000), and "
            "the complete prepared sequence (text, task, prompt, and labels) is limited to 7,999 subword "
            "tokens without silent truncation.\n\n"
            "### Results\n\n"
            "`choice` is the highest-scoring candidate label, and `probabilities` contains the native "
            "softmax score distribution across all candidate labels in request order. "
            "Scores are not calibrated probabilities of correctness."
        ),
        lifespan=lifespan,
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    )
    app.state.runtime = active_runtime
    app.add_middleware(_PublicClassifyLimitMiddleware, max_bytes=MAX_PUBLIC_BODY_BYTES)
    public_limiter = anyio.CapacityLimiter(PUBLIC_CLASSIFY_CONCURRENCY)

    @app.exception_handler(RecursionError)
    async def handle_recursion_error(_request: Request, _exc: RecursionError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "Input nesting depth exceeds limit."})

    bearer = HTTPBearer(
        auto_error=False,
        scheme_name="BearerKey",
        description="Any key configured in API_KEYS (or the legacy API_KEY setting)",
    )

    async def authenticate(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        valid_key = False
        if credentials is not None:
            provided_key = credentials.credentials.encode()
            # Compare every configured key without stopping after a match.
            for api_key in api_keys:
                valid_key |= secrets.compare_digest(provided_key, api_key)
        if not valid_key:
            raise HTTPException(401, "Invalid or missing Bearer key", headers={"WWW-Authenticate": "Bearer"})

    async def require_accepting_requests():
        if not active_runtime.snapshot().accept_requests:
            raise HTTPException(503, "Service is temporarily paused for maintenance.")

    @app.get("/", include_in_schema=False)
    async def homepage():
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    @app.get("/api/healthz", tags=["Server"], summary="Server readiness and model residency")
    async def healthz():
        return {"ok": True, "model_loaded": app.state.worker.is_loaded(), "model": MODEL_ID}

    @app.post(
        "/api/classify",
        response_model=ClassifyResponse,
        dependencies=[Depends(authenticate), Depends(require_accepting_requests)],
        tags=["Classification"],
        summary="Classify text against competing candidate labels",
        responses={
            401: {"description": "Missing or invalid Bearer key"},
            413: {"description": "Request body exceeds 128 KiB limit"},
            422: {"description": "Invalid input or prepared sequence exceeds 7,999 tokens"},
            503: {"description": "Service is temporarily paused for maintenance or worker is unavailable"},
        },
    )
    async def classify(request: Request, body: ClassifyRequest):
        settings = active_runtime.snapshot()
        if not settings.accept_requests:
            raise HTTPException(503, "Service is temporarily paused for maintenance.")
        if len(body.labels) > settings.max_labels:
            raise HTTPException(
                422,
                f"labels must contain at most {settings.max_labels} candidate labels.",
            )
        text = body.rendered_text
        if len(text) > settings.max_text_chars:
            raise HTTPException(
                422,
                f"text must not exceed {settings.max_text_chars:,} characters after serialization.",
            )

        cancel_event = threading.Event()
        classify_fn = app.state.worker.classify
        kwargs: dict[str, Any] = {}
        if _supports_kwarg(classify_fn, "cancel_event"):
            kwargs["cancel_event"] = cancel_event
        if _supports_kwarg(classify_fn, "plane"):
            kwargs["plane"] = "interactive"

        def _invoke() -> tuple[str, dict[str, float]]:
            return classify_fn(text, body.labels, body.task, body.prompt, **kwargs)

        result: tuple[str, dict[str, float]] | None = None
        worker_exc: Exception | None = None
        disconnected = False
        try:
            async with anyio.create_task_group() as tg:
                async def _watch_disconnect() -> None:
                    nonlocal disconnected
                    while True:
                        try:
                            msg = await request.receive()
                        except Exception:
                            return
                        if msg.get("type") == "http.disconnect":
                            disconnected = True
                            cancel_event.set()
                            tg.cancel_scope.cancel()
                            return
                        if not msg.get("more_body", False):
                            return

                tg.start_soon(_watch_disconnect)
                try:
                    result = await anyio.to_thread.run_sync(
                        _invoke,
                        limiter=public_limiter,
                        abandon_on_cancel=True,
                    )
                except Exception as exc:
                    worker_exc = exc
                finally:
                    tg.cancel_scope.cancel()
        finally:
            cancel_event.set()

        if worker_exc is not None:
            if isinstance(worker_exc, WorkerInputTooLongError):
                raise HTTPException(422, str(worker_exc)) from worker_exc
            if isinstance(worker_exc, WorkerCancelledError):
                return JSONResponse(status_code=499, content={"detail": "Request cancelled."})
            if isinstance(worker_exc, RuntimeError):
                logging.warning("Public classification failed due to worker error: %s", worker_exc)
                raise HTTPException(503, "Model worker is temporarily unavailable.") from worker_exc
            raise worker_exc

        if disconnected or result is None:
            return JSONResponse(status_code=499, content={"detail": "Client disconnected."})

        choice, probabilities = result
        return ClassifyResponse(choice=choice, probabilities=probabilities)

    return app
