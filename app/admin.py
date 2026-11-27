"""Internal-only Admin API (:9090) for choice.cubus.sh."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import dataclass, field
import hmac
import inspect
import ipaddress
import json
import logging
import os
import socket
import sys
import threading
from typing import Any

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from .api_keys import (
    APIKeyStore,
    APIKeyStoreError,
    APIKeyValidationError,
    parse_api_key_uuid,
    validate_create_api_key_payload,
)
from .runtime import (
    APP_VERSION,
    MAIN_PORT,
    MODEL_ID,
    MODEL_IDS,
    SETTINGS_SCHEMA,
    SUPPORTED_MODELS,
    ChoiceRuntime,
    SettingsValidationError,
    resolve_model_id,
)
from .schemas import build_classification_result, render_text, unpack_classify_payload
from .worker import WorkerCancelledError


def _supports_kwarg(fn: Any, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


DEFAULT_ADMIN_PORT = 9090
MIN_ADMIN_TOKEN_LEN = 16
MAX_ADMIN_BODY_BYTES = 16 << 10  # 16 KiB
MAX_ADMIN_CLASSIFY_BODY_BYTES = 256 << 10  # 256 KiB
MAX_ADMIN_BATCH_ITEMS = 64
MAX_ADMIN_ITEM_TEXT_CHARS = 16000
MAX_ADMIN_PROMPT_CHARS = 512
MAX_ADMIN_TASK_CHARS = 64
MIN_ADMIN_LABELS = 2
MAX_ADMIN_LABELS = 16
MAX_ADMIN_LABEL_NAME_CHARS = 64
MAX_ADMIN_LABEL_DESC_CHARS = 240

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


class _SockAddr(ctypes.Structure):
    _fields_ = [
        ("sa_family", ctypes.c_ushort),
        ("sa_data", ctypes.c_char * 14),
    ]


class _SockAddrIn(ctypes.Structure):
    _fields_ = [
        ("sin_family", ctypes.c_ushort),
        ("sin_port", ctypes.c_uint16),
        ("sin_addr", ctypes.c_ubyte * 4),
        ("sin_zero", ctypes.c_char * 8),
    ]


class _SockAddrIn6(ctypes.Structure):
    _fields_ = [
        ("sin6_family", ctypes.c_ushort),
        ("sin6_port", ctypes.c_uint16),
        ("sin6_flowinfo", ctypes.c_uint32),
        ("sin6_addr", ctypes.c_ubyte * 16),
        ("sin6_scope_id", ctypes.c_uint32),
    ]


class _IfAddrs(ctypes.Structure):
    pass


_IfAddrs._fields_ = [
    ("ifa_next", ctypes.POINTER(_IfAddrs)),
    ("ifa_name", ctypes.c_char_p),
    ("ifa_flags", ctypes.c_uint),
    ("ifa_addr", ctypes.POINTER(_SockAddr)),
    ("ifa_netmask", ctypes.POINTER(_SockAddr)),
    ("ifa_ifu", ctypes.c_void_p),
    ("ifa_data", ctypes.c_void_p),
]


def _load_libc() -> ctypes.CDLL:
    for lib_name in ("libc.so.6", None):
        try:
            lib = ctypes.CDLL(lib_name, use_errno=True)
        except OSError:
            continue
        if hasattr(lib, "getifaddrs") and hasattr(lib, "freeifaddrs"):
            lib.getifaddrs.argtypes = [ctypes.POINTER(ctypes.POINTER(_IfAddrs))]
            lib.getifaddrs.restype = ctypes.c_int
            lib.freeifaddrs.argtypes = [ctypes.POINTER(_IfAddrs)]
            lib.freeifaddrs.restype = None
            return lib
    raise RuntimeError("Unable to load libc getifaddrs/freeifaddrs for ADMIN_BIND_CIDR resolution.")


def _unmap_ip(addr: IPAddress) -> IPAddress:
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def list_interface_ips() -> list[IPAddress]:
    """Enumerate assigned local interface IP addresses using POSIX/Linux getifaddrs."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError(f"ADMIN_BIND_CIDR interface resolution is supported on Linux only (got {sys.platform!r}).")

    libc = _load_libc()
    head = ctypes.POINTER(_IfAddrs)()
    if libc.getifaddrs(ctypes.byref(head)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))

    addrs: list[IPAddress] = []
    try:
        cur = head
        while cur:
            ifa = cur.contents
            if ifa.ifa_addr:
                family = ifa.ifa_addr.contents.sa_family
                if family == socket.AF_INET:
                    sa4 = ctypes.cast(ifa.ifa_addr, ctypes.POINTER(_SockAddrIn)).contents
                    ip: IPAddress = ipaddress.IPv4Address(bytes(sa4.sin_addr))
                    if not ip.is_unspecified:
                        addrs.append(ip)
                elif family == socket.AF_INET6:
                    sa6 = ctypes.cast(ifa.ifa_addr, ctypes.POINTER(_SockAddrIn6)).contents
                    ip6 = _unmap_ip(ipaddress.IPv6Address(bytes(sa6.sin6_addr)))
                    if not ip6.is_unspecified and not (isinstance(ip6, ipaddress.IPv6Address) and ip6.is_link_local):
                        addrs.append(ip6)
            cur = ifa.ifa_next
    finally:
        if head:
            libc.freeifaddrs(head)
    return addrs


def _is_placeholder(value: str) -> bool:
    low = value.strip().lower()
    return low.startswith("replace-with-") or low in {"changeme", "change-me"}


def _parse_cidr(raw: str, name: str) -> IPNetwork | None:
    value = raw.strip()
    if not value:
        return None
    if "/" not in value:
        raise ValueError(f"Invalid {name} {raw!r}: expected CIDR notation (e.g. 172.31.0.0/24).")
    try:
        network = ipaddress.ip_network(value, strict=False)
    except ValueError as exc:
        raise ValueError(f"Invalid {name} {raw!r}: {exc}") from exc
    if network.prefixlen == 0:
        raise ValueError(f"{name} {raw!r} must not be a catch-all /0 prefix.")
    if network.network_address.is_unspecified:
        raise ValueError(f"{name} {raw!r} must not use an unspecified network address.")
    if isinstance(network, ipaddress.IPv6Network) and network.network_address.ipv4_mapped is not None:
        raise ValueError(f"{name} {raw!r} must not use an IPv4-mapped IPv6 prefix.")
    return network


@dataclass(frozen=True, slots=True)
class AdminConfig:
    """Validated configuration for the optional internal Admin API."""

    token: str = field(default="", repr=False)
    port: int = DEFAULT_ADMIN_PORT
    bind_cidr: IPNetwork | None = None
    allow_cidr: IPNetwork | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def __repr__(self) -> str:
        token_state = "[redacted]" if self.token else ""
        return (
            f"AdminConfig(enabled={self.enabled}, port={self.port}, "
            f"bind_cidr={self.bind_cidr!r}, allow_cidr={self.allow_cidr!r}, token={token_state!r})"
        )


def load_admin_config(environ: Mapping[str, str] | None = None) -> AdminConfig:
    """Load and validate Admin API environment configuration."""
    env = os.environ if environ is None else environ

    port = DEFAULT_ADMIN_PORT
    raw_port = env.get("ADMIN_PORT", "").strip()
    if raw_port:
        if not raw_port.isascii() or not raw_port.isdigit():
            raise ValueError(f"Invalid ADMIN_PORT {raw_port!r}: must be an integer between 1 and 65535.")
        port = int(raw_port)
        if port < 1 or port > 65535:
            raise ValueError(f"Invalid ADMIN_PORT {raw_port!r}: must be between 1 and 65535.")
        if port == MAIN_PORT:
            raise ValueError(f"Invalid ADMIN_PORT {port}: must differ from main HTTP port {MAIN_PORT}.")

    bind_cidr = _parse_cidr(env.get("ADMIN_BIND_CIDR", ""), "ADMIN_BIND_CIDR")
    allow_cidr = _parse_cidr(env.get("ADMIN_ALLOW_CIDR", ""), "ADMIN_ALLOW_CIDR")
    if allow_cidr is None and bind_cidr is not None:
        allow_cidr = bind_cidr

    token = env.get("ADMIN_API_TOKEN", "").strip()
    if token:
        if _is_placeholder(token):
            raise ValueError("ADMIN_API_TOKEN must not be a placeholder.")
        if not token.isascii() or any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in token):
            raise ValueError("ADMIN_API_TOKEN must contain printable ASCII characters only.")
        if len(token.encode("ascii")) < MIN_ADMIN_TOKEN_LEN:
            raise ValueError(f"ADMIN_API_TOKEN must be at least {MIN_ADMIN_TOKEN_LEN} bytes.")

    return AdminConfig(
        token=token,
        port=port,
        bind_cidr=bind_cidr,
        allow_cidr=allow_cidr,
    )


def resolve_bind_addr(
    bind_cidr: IPNetwork | None,
    port: int,
    interface_ips: Sequence[IPAddress] | None = None,
) -> tuple[str, int]:
    """Resolve the exact bind IP/port (127.0.0.1 when bind_cidr is unset; never 0.0.0.0 or ::)."""
    if port < 1 or port > 65535 or port == MAIN_PORT:
        raise ValueError(f"Invalid admin port {port}.")
    if bind_cidr is None:
        return ("127.0.0.1", port)

    addrs = list_interface_ips() if interface_ips is None else interface_ips
    for raw_addr in addrs:
        addr = _unmap_ip(raw_addr)
        if addr.is_unspecified:
            continue
        if addr in bind_cidr:
            return (str(addr), port)
    raise RuntimeError(f"No local interface IP found inside ADMIN_BIND_CIDR {bind_cidr}.")


def _parse_peer_ip(client: Any) -> IPAddress | None:
    if not isinstance(client, (tuple, list)) or not client:
        return None
    raw_host = client[0]
    if not isinstance(raw_host, str):
        return None
    host = raw_host.strip()
    if not host or "%" in host:
        return None
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return None
    return _unmap_ip(addr)


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"invalid JSON constant {token!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


class _AdminBodyError(ValueError):
    """HTTP-status-tagged request body or JSON parsing error."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _parse_strict_json_object(raw_body: bytes, context: str = "settings patch") -> dict[str, Any]:
    try:
        text = raw_body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SettingsValidationError(f"invalid UTF-8 in {context}") from exc
    if not text.strip():
        raise SettingsValidationError(f"invalid {context} JSON")
    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (ValueError, RecursionError) as exc:
        raise SettingsValidationError(f"invalid {context} JSON") from exc
    if not isinstance(parsed, dict):
        raise SettingsValidationError(f"{context} must be a JSON object")
    return parsed


async def _read_bounded_json_object(request: Request, max_bytes: int, context: str) -> dict[str, Any]:
    cl_values = request.headers.getlist("content-length")
    if len(cl_values) > 1:
        raise _AdminBodyError(400, "invalid Content-Length header")
    if len(cl_values) == 1:
        cl_str = cl_values[0].strip()
        if not cl_str.isascii() or not cl_str.isdigit():
            raise _AdminBodyError(400, "invalid Content-Length header")
        if int(cl_str) > max_bytes:
            raise _AdminBodyError(413, "request body too large")

    body = bytearray()
    try:
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_bytes:
                raise _AdminBodyError(413, "request body too large")
            body.extend(chunk)
    except ClientDisconnect as exc:
        raise _AdminBodyError(499, "request cancelled") from exc

    try:
        return _parse_strict_json_object(bytes(body), context=context)
    except SettingsValidationError as exc:
        raise _AdminBodyError(400, str(exc)) from exc


def _validate_non_blank_str(value: object, name: str, max_len: int) -> str:
    if not isinstance(value, str):
        raise SettingsValidationError(f"{name} must be a string")
    if not value.strip():
        raise SettingsValidationError(f"{name} must not be blank")
    if len(value) > max_len:
        raise SettingsValidationError(f"{name} must not exceed {max_len} characters")
    return value


def _validate_admin_input_item(entry: object, name: str) -> str:
    if isinstance(entry, str):
        rendered = entry
    elif isinstance(entry, dict):
        unknown = sorted(set(entry) - {"text"})
        if unknown:
            raise SettingsValidationError(f"{name} contains unknown field {unknown[0]!r}")
        if "text" not in entry:
            raise SettingsValidationError(f"{name} must contain 'text'")
        raw_text = entry["text"]
        if not isinstance(raw_text, (str, dict, list)):
            raise SettingsValidationError(f"{name}.text must be a string, object, or array")
        try:
            rendered = render_text(raw_text)
        except (ValueError, RecursionError) as exc:
            raise SettingsValidationError(f"{name}.text contains invalid JSON value") from exc
    else:
        raise SettingsValidationError(f"{name} must be a string or an object with 'text'")

    if not rendered.strip():
        raise SettingsValidationError(f"{name} must not be blank")
    if len(rendered) > MAX_ADMIN_ITEM_TEXT_CHARS:
        raise SettingsValidationError(f"{name} must not exceed {MAX_ADMIN_ITEM_TEXT_CHARS} characters")
    return rendered


def _validate_classify_payload(
    parsed: dict[str, Any],
) -> tuple[str, list[str], list[str] | dict[str, str | None], str, str | None]:
    allowed_top = {"model", "input", "labels", "task", "prompt"}
    unknown_top = sorted(set(parsed) - allowed_top)
    if unknown_top:
        raise SettingsValidationError(f"unknown field {unknown_top[0]!r}")
    if "labels" not in parsed:
        raise SettingsValidationError("labels is required")
    if "input" not in parsed:
        raise SettingsValidationError("input is required")

    canonical_model = MODEL_ID
    if "model" in parsed:
        raw_model = parsed["model"]
        if not isinstance(raw_model, str):
            raise SettingsValidationError("model must be a string")
        resolved = resolve_model_id(raw_model)
        if resolved is None:
            supported = ", ".join(sorted(SUPPORTED_MODELS))
            raise SettingsValidationError(f"unknown model {raw_model!r}; supported models: {supported}")
        canonical_model = resolved

    prompt: str | None = None
    if "prompt" in parsed and parsed["prompt"] is not None:
        prompt = _validate_non_blank_str(parsed["prompt"], "prompt", MAX_ADMIN_PROMPT_CHARS)

    task = "decision"
    if "task" in parsed:
        task = _validate_non_blank_str(parsed["task"], "task", MAX_ADMIN_TASK_CHARS)

    raw_labels = parsed["labels"]
    validated_labels: list[str] | dict[str, str | None]
    if isinstance(raw_labels, list):
        if not (MIN_ADMIN_LABELS <= len(raw_labels) <= MAX_ADMIN_LABELS):
            raise SettingsValidationError(
                f"labels must contain between {MIN_ADMIN_LABELS} and {MAX_ADMIN_LABELS} items"
            )
        label_list: list[str] = []
        for idx, raw_name in enumerate(raw_labels):
            name = _validate_non_blank_str(raw_name, f"labels[{idx}]", MAX_ADMIN_LABEL_NAME_CHARS)
            label_list.append(name)
        if len(set(label_list)) != len(label_list):
            raise SettingsValidationError("label names must be unique")
        validated_labels = label_list
    elif isinstance(raw_labels, dict):
        if not (MIN_ADMIN_LABELS <= len(raw_labels) <= MAX_ADMIN_LABELS):
            raise SettingsValidationError(
                f"labels must contain between {MIN_ADMIN_LABELS} and {MAX_ADMIN_LABELS} entries"
            )
        label_dict: dict[str, str | None] = {}
        for raw_name, raw_desc in raw_labels.items():
            name = _validate_non_blank_str(raw_name, "label name", MAX_ADMIN_LABEL_NAME_CHARS)
            if raw_desc is None:
                label_dict[name] = None
            else:
                label_dict[name] = _validate_non_blank_str(
                    raw_desc, f"label description for {name!r}", MAX_ADMIN_LABEL_DESC_CHARS
                )
        validated_labels = label_dict
    else:
        raise SettingsValidationError("labels must be a list of strings or an object")

    raw_input = parsed["input"]
    validated_texts: list[str] = []
    if isinstance(raw_input, list):
        if not (1 <= len(raw_input) <= MAX_ADMIN_BATCH_ITEMS):
            raise SettingsValidationError(f"input must contain between 1 and {MAX_ADMIN_BATCH_ITEMS} entries")
        for idx, entry in enumerate(raw_input):
            validated_texts.append(_validate_admin_input_item(entry, f"input[{idx}]"))
    elif isinstance(raw_input, (str, dict)):
        validated_texts.append(_validate_admin_input_item(raw_input, "input"))
    else:
        raise SettingsValidationError("input must be a string, an object with 'text', or a non-empty list")

    return canonical_model, validated_texts, validated_labels, task, prompt


def _error_response(
    status_code: int,
    message: str,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message}, headers=headers)


async def _send_json_error(
    send: Any,
    method: str,
    status_code: int,
    message: str,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps({"error": message}, separators=(",", ":")).encode("utf-8")
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"cache-control", b"no-store"),
        (b"x-content-type-options", b"nosniff"),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status_code, "headers": headers})
    await send({"type": "http.response.body", "body": b"" if method == "HEAD" else body, "more_body": False})


class _AdminGuardMiddleware:
    """ASGI security guard enforcing peer IP allowlist, constant-time Bearer auth, and response headers."""

    def __init__(self, app: Any, config: AdminConfig) -> None:
        self._app = app
        self._allow_cidr = config.allow_cidr
        self._token_bytes = config.token.encode("ascii")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await self._app(scope, receive, send)
            return
        if scope_type != "http":
            if scope_type == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return

        method = str(scope.get("method", "GET")).upper()
        peer_ip = _parse_peer_ip(scope.get("client"))
        if peer_ip is None:
            await _send_json_error(send, method, 403, "invalid peer address")
            return

        if self._allow_cidr is not None:
            if peer_ip not in self._allow_cidr:
                await _send_json_error(send, method, 403, "peer outside allowed admin subnet")
                return
        elif not peer_ip.is_loopback:
            await _send_json_error(send, method, 403, "admin API restricted to loopback")
            return

        auth_headers = [val for name, val in scope.get("headers", ()) if name.lower() == b"authorization"]
        if len(auth_headers) != 1:
            await _send_json_error(
                send,
                method,
                401,
                "missing or invalid admin bearer token",
                extra_headers=[(b"www-authenticate", b'Bearer realm="choice-admin"')],
            )
            return

        parts = auth_headers[0].split()
        if len(parts) != 2 or parts[0].lower() != b"bearer" or not hmac.compare_digest(
            parts[1], self._token_bytes
        ):
            await _send_json_error(
                send,
                method,
                401,
                "missing or invalid admin bearer token",
                extra_headers=[(b"www-authenticate", b'Bearer realm="choice-admin"')],
            )
            return

        async def send_with_security_headers(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                filtered = [
                    (k, v)
                    for k, v in message.get("headers", ())
                    if k.lower() not in (b"cache-control", b"x-content-type-options")
                ]
                filtered.append((b"cache-control", b"no-store"))
                filtered.append((b"x-content-type-options", b"nosniff"))
                message = {**message, "headers": filtered}
            await send(message)

        await self._app(scope, receive, send_with_security_headers)


def create_admin_app(
    runtime: ChoiceRuntime,
    config: AdminConfig,
    api_key_store: APIKeyStore | None = None,
) -> Any:
    """Build the isolated internal Admin API ASGI application."""
    if not config.enabled:
        raise ValueError("Admin API cannot be created without a validated ADMIN_API_TOKEN.")

    app = FastAPI(
        title="Choice Internal Admin API",
        version=APP_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    # Separate limiters keep :8000 saturation and heavy :9090 batch inference from starving emergency settings or key ops.
    settings_limiter = anyio.CapacityLimiter(4)
    classify_limiter = anyio.CapacityLimiter(4)
    api_keys_limiter = anyio.CapacityLimiter(4)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(exc.status_code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(_request: Request, _exc: Exception) -> JSONResponse:
        return _error_response(500, "internal server error")

    @app.get("/admin/v1/manifest")
    async def get_manifest() -> dict[str, Any]:
        return {
            "service": "choice",
            "version": APP_VERSION,
            "model": MODEL_ID,
            "models": list(MODEL_IDS),
            "capabilities": ["settings", "classify", "api_keys"],
            "endpoints": {
                "settings_get": "GET /admin/v1/settings",
                "settings_update": "PATCH /admin/v1/settings",
                "classify": "POST /admin/v1/classify",
                "api_keys_list": "GET /admin/v1/api-keys",
                "api_keys_create": "POST /admin/v1/api-keys",
                "api_keys_revoke": "POST /admin/v1/api-keys/{id}/revoke",
                "api_keys_delete": "DELETE /admin/v1/api-keys/{id}",
            },
            "settings_schema": [dict(item) for item in SETTINGS_SCHEMA],
        }

    @app.get("/admin/v1/api-keys")
    async def list_api_keys() -> JSONResponse:
        if api_key_store is None:
            return _error_response(503, "api key store unavailable")
        try:
            records = await anyio.to_thread.run_sync(api_key_store.list_keys, limiter=api_keys_limiter)
        except APIKeyStoreError:
            return _error_response(503, "api key store unavailable")
        return JSONResponse(status_code=200, content={"api_keys": records})

    @app.post("/admin/v1/api-keys")
    async def create_api_key(request: Request) -> JSONResponse:
        try:
            parsed = await _read_bounded_json_object(request, MAX_ADMIN_BODY_BYTES, "api key create")
            name, owner, expires_in_days, created_by = validate_create_api_key_payload(parsed)
        except _AdminBodyError as exc:
            return _error_response(exc.status_code, str(exc))
        except APIKeyValidationError as exc:
            return _error_response(400, str(exc))

        if api_key_store is None:
            return _error_response(503, "api key store unavailable")

        def _create() -> tuple[str, dict[str, Any]]:
            return api_key_store.create_key(
                name=name,
                owner=owner,
                expires_in_days=expires_in_days,
                created_by=created_by,
            )

        try:
            plaintext, record = await anyio.to_thread.run_sync(_create, limiter=api_keys_limiter)
        except APIKeyStoreError:
            return _error_response(503, "api key store unavailable")

        return JSONResponse(status_code=201, content={"api_key": plaintext, "record": record})

    @app.post("/admin/v1/api-keys/{key_id}/revoke")
    async def revoke_api_key(key_id: str, request: Request) -> JSONResponse:
        cl_values = request.headers.getlist("content-length")
        if len(cl_values) > 1:
            return _error_response(400, "invalid Content-Length header")
        if len(cl_values) == 1:
            cl_str = cl_values[0].strip()
            if not cl_str.isascii() or not cl_str.isdigit():
                return _error_response(400, "invalid Content-Length header")
            if int(cl_str) > MAX_ADMIN_BODY_BYTES:
                return _error_response(413, "request body too large")

        try:
            parsed_id = parse_api_key_uuid(key_id)
        except APIKeyValidationError as exc:
            return _error_response(400, str(exc))

        if api_key_store is None:
            return _error_response(503, "api key store unavailable")

        try:
            updated = await anyio.to_thread.run_sync(api_key_store.revoke_key, parsed_id, limiter=api_keys_limiter)
        except APIKeyStoreError:
            return _error_response(503, "api key store unavailable")

        if updated is None:
            return _error_response(404, "api key not found")
        return JSONResponse(status_code=200, content=updated)

    @app.delete("/admin/v1/api-keys/{key_id}")
    async def delete_api_key(key_id: str) -> JSONResponse:
        try:
            parsed_id = parse_api_key_uuid(key_id)
        except APIKeyValidationError as exc:
            return _error_response(400, str(exc))

        if api_key_store is None:
            return _error_response(503, "api key store unavailable")

        try:
            deleted = await anyio.to_thread.run_sync(api_key_store.delete_key, parsed_id, limiter=api_keys_limiter)
        except APIKeyStoreError:
            return _error_response(503, "api key store unavailable")

        if not deleted:
            return _error_response(404, "api key not found")
        return JSONResponse(status_code=200, content={"deleted": True, "id": str(parsed_id)})

    @app.get("/admin/v1/settings")
    async def get_settings() -> dict[str, bool | int]:
        return runtime.to_dict()

    @app.patch("/admin/v1/settings")
    async def patch_settings(request: Request) -> JSONResponse:
        try:
            parsed = await _read_bounded_json_object(request, MAX_ADMIN_BODY_BYTES, "settings patch")
            updated = await anyio.to_thread.run_sync(runtime.apply_patch, parsed, limiter=settings_limiter)
        except _AdminBodyError as exc:
            return _error_response(exc.status_code, str(exc))
        except SettingsValidationError as exc:
            return _error_response(400, str(exc))

        return JSONResponse(status_code=200, content=updated)

    @app.post("/admin/v1/classify")
    async def classify_admin(request: Request) -> JSONResponse:
        try:
            parsed = await _read_bounded_json_object(
                request, MAX_ADMIN_CLASSIFY_BODY_BYTES, "classify payload"
            )
            canonical_model, validated_texts, labels, task, prompt = _validate_classify_payload(parsed)
        except _AdminBodyError as exc:
            return _error_response(exc.status_code, str(exc))
        except SettingsValidationError as exc:
            return _error_response(400, str(exc))

        worker = runtime.get_worker()
        if worker is None:
            return _error_response(503, "model worker unavailable")

        batch_inputs = [(item_text, labels, task, prompt) for item_text in validated_texts]
        cancel_event = threading.Event()
        classify_batch_fn = worker.classify_batch
        kwargs: dict[str, Any] = {}
        if _supports_kwarg(classify_batch_fn, "model_id"):
            kwargs["model_id"] = canonical_model
        if _supports_kwarg(classify_batch_fn, "cancel_event"):
            kwargs["cancel_event"] = cancel_event
        if _supports_kwarg(classify_batch_fn, "plane"):
            kwargs["plane"] = "admin"

        def _invoke() -> list[tuple[str, Any]]:
            return classify_batch_fn(batch_inputs, **kwargs)

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
                        limiter=classify_limiter,
                        abandon_on_cancel=True,
                    )
                except Exception as exc:
                    worker_exc = exc
                finally:
                    tg.cancel_scope.cancel()
        finally:
            cancel_event.set()

        if worker_exc is not None:
            if isinstance(worker_exc, WorkerCancelledError):
                return _error_response(499, "request cancelled")
            if isinstance(worker_exc, RuntimeError):
                logging.warning("Admin classification failed: %s", worker_exc)
                return _error_response(503, "model worker unavailable")
            raise worker_exc

        if disconnected or raw_results is None:
            return _error_response(499, "request cancelled")

        if len(raw_results) != len(validated_texts):
            logging.error(
                "Admin classification result length mismatch: %d != %d",
                len(raw_results),
                len(validated_texts),
            )
            return _error_response(500, "internal server error")

        data: list[dict[str, Any]] = []
        total_tokens = 0
        for idx, entry_result in enumerate(raw_results):
            status, payload = entry_result
            if status == "ok":
                try:
                    _choice, probabilities, token_count = unpack_classify_payload(payload)
                    item = build_classification_result(idx, probabilities).model_dump()
                except ValueError as exc:
                    data.append({"object": "classification", "index": idx, "error": str(exc)})
                    continue
                data.append(item)
                total_tokens += token_count
            else:
                data.append({"object": "classification", "index": idx, "error": str(payload)})

        return JSONResponse(
            status_code=200,
            content={
                "object": "list",
                "model": canonical_model,
                "data": data,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens,
                },
            },
        )

    return _AdminGuardMiddleware(app, config)
