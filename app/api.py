"""Industry-standard Jina AI / OpenAI-style multi-model classification API."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import inspect
import json
import logging
import os
from pathlib import Path
import threading
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api_keys import APIKeyStore, APIKeyStoreError
from .db import create_postgres_api_key_store
from .runtime import (
    APP_VERSION,
    MODEL_ID,
    MODEL_IDS,
    MODEL_SPECS,
    SUPPORTED_MODELS,
    ChoiceRuntime,
    resolve_model_id,
    resolve_model_paths,
    should_validate_all_model_paths,
)
from .schemas import (
    ClassificationResult,
    ClassifyRequest,
    ClassifyResponse,
    ErrorEnvelope,
    ModelCard,
    ModelListResponse,
    UsageInfo,
    build_classification_result,
    unpack_classify_payload,
)
from .worker import ModelWorker, WorkerCancelledError, WorkerInputTooLongError


def _supports_kwarg(fn: Any, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


MAX_PUBLIC_BODY_BYTES = 256 << 10  # 256 KiB (supports up to 64 batch items)
PUBLIC_CLASSIFY_CONCURRENCY = 24
AUTH_DB_CONCURRENCY = 5  # Leaves headroom inside max_size=8 pool for :9090 Admin API & usage flusher.

_BEARER_SCHEME = HTTPBearer(
    auto_error=False,
    scheme_name="BearerKey",
    description="PostgreSQL-backed API key sent as `Authorization: Bearer <key>`",
)
_API_KEY_HEADER_SCHEME = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    scheme_name="ApiKeyHeader",
    description="PostgreSQL-backed API key sent in the `X-API-Key` header",
)

_STATUS_ERROR_META: dict[int, tuple[str, str]] = {
    400: ("invalid_request_error", "invalid_request"),
    401: ("authentication_error", "invalid_api_key"),
    403: ("permission_error", "forbidden"),
    404: ("invalid_request_error", "not_found"),
    405: ("invalid_request_error", "method_not_allowed"),
    413: ("invalid_request_error", "payload_too_large"),
    422: ("invalid_request_error", "validation_error"),
    499: ("request_cancelled", "request_cancelled"),
    503: ("server_error", "service_unavailable"),
}


def _openai_error_payload(message: str, error_type: str, code: str) -> dict[str, dict[str, str]]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }


def _openai_error_response(
    status_code: int,
    message: str,
    error_type: str,
    code: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=_openai_error_payload(message, error_type, code),
        headers=headers,
    )


def _is_v1_path(path: str) -> bool:
    return path == "/v1" or path.startswith("/v1/")


async def _send_public_json_error(
    send: Any,
    status_code: int,
    message: str,
    error_type: str,
    code: str,
) -> None:
    body = json.dumps(_openai_error_payload(message, error_type, code), separators=(",", ":")).encode("utf-8")
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
    """Enforce a strict body-size cap on POST /v1/classify before JSON parsing."""

    def __init__(self, app: Any, max_bytes: int = MAX_PUBLIC_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or str(scope.get("method", "")).upper() != "POST"
            or scope.get("path") != "/v1/classify"
        ):
            await self.app(scope, receive, send)
            return

        cl_headers = [val for name, val in scope.get("headers", ()) if name.lower() == b"content-length"]
        if len(cl_headers) > 1:
            await _send_public_json_error(
                send,
                400,
                "Invalid Content-Length header.",
                "invalid_request_error",
                "invalid_content_length",
            )
            return
        if len(cl_headers) == 1:
            cl_str = cl_headers[0].decode("latin-1").strip()
            if not cl_str.isascii() or not cl_str.isdigit():
                await _send_public_json_error(
                    send,
                    400,
                    "Invalid Content-Length header.",
                    "invalid_request_error",
                    "invalid_content_length",
                )
                return
            if int(cl_str) > self.max_bytes:
                await _send_public_json_error(
                    send,
                    413,
                    "Request body too large.",
                    "invalid_request_error",
                    "payload_too_large",
                )
                return

        buf = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            msg_type = message.get("type")
            if msg_type == "http.disconnect":
                await _send_public_json_error(
                    send,
                    499,
                    "Client disconnected.",
                    "request_cancelled",
                    "request_cancelled",
                )
                return
            if msg_type == "http.request":
                chunk = message.get("body", b"")
                if len(buf) + len(chunk) > self.max_bytes:
                    await _send_public_json_error(
                        send,
                        413,
                        "Request body too large.",
                        "invalid_request_error",
                        "payload_too_large",
                    )
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


def _extract_api_key(request: Request) -> str | None:
    """Extract a single unambiguous API key from Authorization: Bearer or X-API-Key."""
    auth_values = request.headers.getlist("authorization")
    x_api_key_values = request.headers.getlist("x-api-key")

    if len(auth_values) > 1 or len(x_api_key_values) > 1:
        return None

    bearer_token: str | None = None
    if len(auth_values) == 1:
        parts = auth_values[0].split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        bearer_token = parts[1]

    header_token: str | None = None
    if len(x_api_key_values) == 1:
        raw = x_api_key_values[0].strip()
        if not raw or any(ch.isspace() for ch in raw):
            return None
        header_token = raw

    if bearer_token is not None and header_token is not None:
        try:
            bearer_bytes = bearer_token.encode("utf-8", errors="strict")
            header_bytes = header_token.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            return None
        if not hmac.compare_digest(bearer_bytes, header_bytes):
            return None
        return bearer_token

    return bearer_token if bearer_token is not None else header_token


def create_app(
    runtime: ChoiceRuntime | None = None,
    api_key_store: APIKeyStore | None = None,
):
    active_runtime = ChoiceRuntime.from_env() if runtime is None else runtime

    @asynccontextmanager
    async def lifespan(app):
        owns_store = api_key_store is None
        store = (
            await anyio.to_thread.run_sync(create_postgres_api_key_store)
            if api_key_store is None
            else api_key_store
        )
        store.start_background_flush()
        app.state.api_key_store = store

        device = os.environ.get("DEVICE", "cuda")
        precision = os.environ.get("PRECISION")
        idle_timeout = active_runtime.initial_worker_idle_timeout()
        model_paths = resolve_model_paths()
        worker_model_arg: str | dict[str, str] = (
            model_paths if should_validate_all_model_paths() else model_paths[MODEL_ID]
        )
        try:
            worker = ModelWorker(
                worker_model_arg,
                model_paths=model_paths,
                device=device,
                precision=precision,
                idle_timeout=idle_timeout,
            )
        except BaseException:
            if owns_store:
                await anyio.to_thread.run_sync(store.close)
            else:
                await anyio.to_thread.run_sync(store.stop_background_flush)
            raise

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
                try:
                    await anyio.to_thread.run_sync(worker.close)
                finally:
                    if owns_store:
                        await anyio.to_thread.run_sync(store.close)
                    else:
                        await anyio.to_thread.run_sync(store.stop_background_flush)

    app = FastAPI(
        title="Choice API",
        version=APP_VERSION,
        description=(
            "Classify single or batched text inputs against candidate labels using Amber 150M "
            "(`MLukauskas/amber-150m`, fast English-primary) or BGE-M3 Zero-Shot v2.0 "
            "(`MoritzLaurer/bge-m3-zeroshot-v2.0`, multilingual/Korean NLI up to 8,192 tokens) "
            "via an industry-standard Jina AI / OpenAI-style `/v1` API.\n\n"
            "### Writing requests\n\n"
            "Send `POST /v1/classify` with `input` (a single string, a `{\"text\": ...}` object, or a "
            "batch list of 1–64 strings and/or `{\"text\": ...}` objects) and `labels` (2–16 competing "
            "candidate labels as a list of strings or a label-to-description mapping). Optionally set "
            "`model` (`amber-150m`, `MLukauskas/amber-150m`, `bge-m3-zeroshot-v2.0`, or "
            "`MoritzLaurer/bge-m3-zeroshot-v2.0`), `task` (defaults to `decision`), and "
            "`prompt` (a question or classification instruction).\n\n"
            "- For binary or yes/no questions, supply two meaningful competing labels (such as "
            '`["no", "yes"]` or `{"safe": "Read-only inspection", "unsafe": "Modifies or deletes state"}`).\n'
            "- Include concise label descriptions when category boundaries are domain-specific.\n"
            "- Each serialized input item defaults to an 8,000-character limit (configurable up to 16,000). "
            "Token limits (7,999 tokens for `amber-150m`; 8,192 tokens per pair for `bge-m3-zeroshot-v2.0`) "
            "are enforced without silent truncation.\n\n"
            "### Results\n\n"
            "Each entry in `data` contains `prediction` (winning label), `score` (winning softmax score), "
            "and `predictions` sorted descending by score (ties preserve requested label order). "
            "`usage` reports exact `prompt_tokens` and `total_tokens` across the batch. "
            "Scores are not calibrated probabilities of correctness."
        ),
        lifespan=lifespan,
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    )
    app.state.runtime = active_runtime
    app.state.api_key_store = api_key_store
    app.add_middleware(_PublicClassifyLimitMiddleware, max_bytes=MAX_PUBLIC_BODY_BYTES)
    public_limiter = anyio.CapacityLimiter(PUBLIC_CLASSIFY_CONCURRENCY)
    auth_db_limiter = anyio.CapacityLimiter(AUTH_DB_CONCURRENCY)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if _is_v1_path(request.url.path):
            if isinstance(exc.detail, dict) and "error" in exc.detail:
                return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=exc.headers)
            default_meta = (
                ("server_error", "internal_error")
                if exc.status_code >= 500
                else ("invalid_request_error", "invalid_request")
            )
            err_type, err_code = _STATUS_ERROR_META.get(exc.status_code, default_meta)
            return _openai_error_response(
                exc.status_code,
                str(exc.detail),
                err_type,
                err_code,
                headers=dict(exc.headers) if exc.headers else None,
            )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        if _is_v1_path(request.url.path):
            parts: list[str] = []
            for err in exc.errors():
                loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
                msg = str(err.get("msg", "Invalid value"))
                parts.append(f"{loc}: {msg}" if loc else msg)
            message = "; ".join(parts) if parts else "Invalid request payload."
            return _openai_error_response(422, message, "invalid_request_error", "validation_error")
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(RecursionError)
    async def handle_recursion_error(request: Request, _exc: RecursionError) -> JSONResponse:
        if _is_v1_path(request.url.path):
            return _openai_error_response(
                422,
                "Input nesting depth exceeds limit.",
                "invalid_request_error",
                "nesting_depth_exceeded",
            )
        return JSONResponse(status_code=422, content={"detail": "Input nesting depth exceeds limit."})

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        logging.exception("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
        if _is_v1_path(request.url.path):
            return _openai_error_response(
                500,
                "Internal server error.",
                "server_error",
                "internal_error",
            )
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    async def authenticate(
        request: Request,
        _credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER_SCHEME)] = None,
        _x_api_key: Annotated[str | None, Depends(_API_KEY_HEADER_SCHEME)] = None,
    ):
        if getattr(request.state, "_authenticated", False):
            return

        raw_key = _extract_api_key(request)
        if raw_key is None:
            raise HTTPException(
                401,
                _openai_error_payload("Invalid or missing API key.", "authentication_error", "invalid_api_key"),
                headers={"WWW-Authenticate": "Bearer"},
            )

        store: APIKeyStore | None = getattr(app.state, "api_key_store", None)
        if store is None:
            raise HTTPException(
                503,
                _openai_error_payload(
                    "Authentication service is temporarily unavailable.",
                    "server_error",
                    "service_unavailable",
                ),
            )

        state, payload = store.verify_key_fast(raw_key)
        if state == "hit":
            request.state._authenticated = True
            return
        if state == "invalid":
            raise HTTPException(
                401,
                _openai_error_payload("Invalid or missing API key.", "authentication_error", "invalid_api_key"),
                headers={"WWW-Authenticate": "Bearer"},
            )

        key_hash, gen_before = payload
        try:
            key_id = await anyio.to_thread.run_sync(
                store.verify_key_miss,
                key_hash,
                gen_before,
                limiter=auth_db_limiter,
            )
        except APIKeyStoreError as exc:
            raise HTTPException(
                503,
                _openai_error_payload(
                    "Authentication service is temporarily unavailable.",
                    "server_error",
                    "service_unavailable",
                ),
            ) from exc

        if key_id is None:
            raise HTTPException(
                401,
                _openai_error_payload("Invalid or missing API key.", "authentication_error", "invalid_api_key"),
                headers={"WWW-Authenticate": "Bearer"},
            )
        request.state._authenticated = True

    async def require_accepting_requests():
        if not active_runtime.snapshot().accept_requests:
            raise HTTPException(
                503,
                _openai_error_payload(
                    "Service is temporarily paused for maintenance.",
                    "server_error",
                    "service_unavailable",
                ),
            )

    class _PreBodyGuardRoute(APIRoute):
        """Enforce API key authentication and maintenance mode before FastAPI decodes JSON bodies."""

        def get_route_handler(self):
            original_handler = super().get_route_handler()

            async def guarded_handler(request: Request):
                await authenticate(request)
                await require_accepting_requests()
                return await original_handler(request)

            return guarded_handler

    v1_classify_router = APIRouter(route_class=_PreBodyGuardRoute)

    @app.get("/", include_in_schema=False)
    async def homepage():
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    @app.get("/healthz", tags=["Server"], summary="Server readiness and model residency")
    async def healthz():
        worker = app.state.worker
        loaded_model_fn = getattr(worker, "loaded_model", None)
        if callable(loaded_model_fn):
            active_model = loaded_model_fn()
            model_loaded = active_model is not None
            model = active_model or MODEL_ID
        else:
            model_loaded = worker.is_loaded()
            model = MODEL_ID
        return {
            "ok": True,
            "model_loaded": model_loaded,
            "model": model,
            "models": list(MODEL_IDS),
        }

    @app.get(
        "/v1/models",
        response_model=ModelListResponse,
        dependencies=[Depends(authenticate)],
        tags=["Models"],
        summary="List available classification models",
        responses={
            401: {"model": ErrorEnvelope, "description": "Missing, invalid, revoked, or expired API key"},
            503: {"model": ErrorEnvelope, "description": "Authentication service is temporarily unavailable"},
        },
    )
    async def list_models() -> ModelListResponse:
        return ModelListResponse(
            object="list",
            data=[ModelCard(id=spec.id, object="model", owned_by=spec.owned_by) for spec in MODEL_SPECS],
        )

    @v1_classify_router.post(
        "/v1/classify",
        response_model=ClassifyResponse,
        dependencies=[Depends(authenticate), Depends(require_accepting_requests)],
        tags=["Classification"],
        summary="Classify single or batched inputs against competing candidate labels",
        responses={
            400: {"model": ErrorEnvelope, "description": "Unknown model or invalid Content-Length header"},
            401: {"model": ErrorEnvelope, "description": "Missing, invalid, revoked, or expired API key"},
            413: {"model": ErrorEnvelope, "description": "Request body exceeds 256 KiB limit"},
            422: {"model": ErrorEnvelope, "description": "Invalid input or prepared sequence exceeds model token limit"},
            499: {"model": ErrorEnvelope, "description": "Client disconnected or request cancelled"},
            503: {"model": ErrorEnvelope, "description": "Service is temporarily paused for maintenance or backend is unavailable"},
        },
    )
    async def classify(request: Request, body: ClassifyRequest):
        settings = active_runtime.snapshot()
        if not settings.accept_requests:
            return _openai_error_response(
                503,
                "Service is temporarily paused for maintenance.",
                "server_error",
                "service_unavailable",
            )
        canonical_model = resolve_model_id(body.model)
        if canonical_model is None:
            supported = ", ".join(sorted(SUPPORTED_MODELS))
            return _openai_error_response(
                400,
                f"Unknown model {body.model!r}. Supported models: {supported}.",
                "invalid_request_error",
                "model_not_found",
            )
        if len(body.labels) > settings.max_labels:
            return _openai_error_response(
                422,
                f"labels must contain at most {settings.max_labels} candidate labels.",
                "invalid_request_error",
                "too_many_labels",
            )

        texts = body.rendered_inputs
        for idx, text in enumerate(texts):
            if len(text) > settings.max_text_chars:
                return _openai_error_response(
                    422,
                    f"input[{idx}] must not exceed {settings.max_text_chars:,} characters after serialization.",
                    "invalid_request_error",
                    "input_too_long",
                )

        batch_inputs = [(text, body.labels, body.task, body.prompt) for text in texts]
        cancel_event = threading.Event()
        worker = app.state.worker
        classify_batch_fn = getattr(worker, "classify_batch", None)

        if callable(classify_batch_fn):
            kwargs: dict[str, Any] = {}
            if _supports_kwarg(classify_batch_fn, "model_id"):
                kwargs["model_id"] = canonical_model
            if _supports_kwarg(classify_batch_fn, "cancel_event"):
                kwargs["cancel_event"] = cancel_event
            if _supports_kwarg(classify_batch_fn, "plane"):
                kwargs["plane"] = "interactive"

            def _invoke() -> list[tuple[str, Any]]:
                return classify_batch_fn(batch_inputs, **kwargs)
        else:
            classify_fn = worker.classify
            kwargs = {}
            if _supports_kwarg(classify_fn, "model_id"):
                kwargs["model_id"] = canonical_model
            if _supports_kwarg(classify_fn, "cancel_event"):
                kwargs["cancel_event"] = cancel_event
            if _supports_kwarg(classify_fn, "plane"):
                kwargs["plane"] = "interactive"

            def _invoke() -> list[tuple[str, Any]]:
                out: list[tuple[str, Any]] = []
                for idx, text in enumerate(texts):
                    if cancel_event.is_set():
                        raise WorkerCancelledError("Classification request was cancelled.")
                    try:
                        out.append(("ok", classify_fn(text, body.labels, body.task, body.prompt, **kwargs)))
                    except WorkerInputTooLongError as exc:
                        msg = str(exc) if len(texts) == 1 else f"input[{idx}]: {exc}"
                        raise WorkerInputTooLongError(msg) from exc
                return out

        raw_results: list[tuple[str, Any]] | None = None
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
                    raw_results = await anyio.to_thread.run_sync(
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
                return _openai_error_response(
                    422,
                    str(worker_exc),
                    "invalid_request_error",
                    "context_length_exceeded",
                )
            if isinstance(worker_exc, WorkerCancelledError):
                return _openai_error_response(
                    499,
                    "Request cancelled.",
                    "request_cancelled",
                    "request_cancelled",
                )
            if isinstance(worker_exc, RuntimeError):
                logging.warning("Public classification failed due to worker error: %s", worker_exc)
                return _openai_error_response(
                    503,
                    "Model worker is temporarily unavailable.",
                    "server_error",
                    "service_unavailable",
                )
            raise worker_exc

        if disconnected or raw_results is None:
            return _openai_error_response(
                499,
                "Client disconnected.",
                "request_cancelled",
                "request_cancelled",
            )

        if len(raw_results) != len(texts):
            logging.error(
                "Public classification result length mismatch: %d != %d",
                len(raw_results),
                len(texts),
            )
            return _openai_error_response(
                503,
                "Model worker is temporarily unavailable.",
                "server_error",
                "service_unavailable",
            )

        for idx, (status, payload) in enumerate(raw_results):
            if status == "input_too_long":
                msg = str(payload) if len(texts) == 1 else f"input[{idx}]: {payload}"
                return _openai_error_response(
                    422,
                    msg,
                    "invalid_request_error",
                    "context_length_exceeded",
                )

        data: list[ClassificationResult] = []
        total_tokens = 0
        for idx, (status, payload) in enumerate(raw_results):
            if status != "ok":
                logging.warning("Public classification item %d failed: %s", idx, payload)
                return _openai_error_response(
                    503,
                    "Model worker is temporarily unavailable.",
                    "server_error",
                    "service_unavailable",
                )
            try:
                _choice, probabilities, token_count = unpack_classify_payload(payload)
                item = build_classification_result(idx, probabilities)
            except ValueError as exc:
                logging.warning("Public classification item %d returned invalid payload: %s", idx, exc)
                return _openai_error_response(
                    503,
                    "Model worker is temporarily unavailable.",
                    "server_error",
                    "service_unavailable",
                )
            data.append(item)
            total_tokens += token_count

        return ClassifyResponse(
            object="list",
            model=canonical_model,
            data=data,
            usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )

    app.include_router(v1_classify_router)
    return app
