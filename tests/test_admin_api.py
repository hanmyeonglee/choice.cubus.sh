"""Unit and integration tests for choice.cubus.sh internal Admin API and live runtime settings."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import ipaddress
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx

from app.__main__ import build_servers, main as app_main, serve_servers
from app.admin import (
    MAX_ADMIN_BODY_BYTES,
    MAX_ADMIN_CLASSIFY_BODY_BYTES,
    AdminConfig,
    create_admin_app,
    list_interface_ips,
    load_admin_config,
    resolve_bind_addr,
)
from app.api import create_app
from app.api_keys import APIKeyStore, InMemoryAPIKeyBackend
from app.db import DatabaseConfigError, resolve_database_config
from app.runtime import APP_VERSION, BGE_M3_MODEL_ID, MODEL_ID, MODEL_IDS, SETTINGS_SCHEMA, ChoiceRuntime
from app.worker import ModelWorker, WorkerInputTooLongError


VALID_TOKEN = "choice-admin-secret-token-123456"


def _make_store_with_key() -> tuple[APIKeyStore, str]:
    store = APIKeyStore(InMemoryAPIKeyBackend())
    plaintext, _record = store.create_key(name="admin-test-key", owner="tests", created_by="suite")
    return store, plaintext


class _FakeWorker:
    """In-memory ModelWorker stand-in that avoids importing torch/transformers in unit tests."""

    def __init__(self, idle_timeout: float = 180.0, *, return_tokens: bool = True) -> None:
        self.idle_timeout = float(idle_timeout)
        self.loaded = False
        self.return_tokens = return_tokens
        self.timeout_updates: list[float] = []
        self.calls: list[dict[str, object]] = []
        self.batch_calls: list[list[tuple[object, ...]]] = []
        self.fail_next_batch_exc: Exception | None = None
        self.score_override: dict[str, float] | None = None

    def is_loaded(self) -> bool:
        return self.loaded

    def start(self) -> None:
        self.loaded = True

    def set_idle_timeout(self, idle_timeout: float) -> None:
        self.idle_timeout = float(idle_timeout)
        self.timeout_updates.append(self.idle_timeout)

    def classify_batch(self, items):
        if self.fail_next_batch_exc is not None:
            exc = self.fail_next_batch_exc
            self.fail_next_batch_exc = None
            raise exc
        self.loaded = True
        self.batch_calls.append(list(items))
        results = []
        for text, labels, task, prompt in items:
            self.calls.append({"text": text, "labels": labels, "task": task, "prompt": prompt})
            if isinstance(text, str) and "__TOO_LONG__" in text:
                results.append(("input_too_long", "Prepared input length (8100 tokens) exceeds the 7,999-token limit."))
                continue
            if isinstance(text, str) and "__ERROR__" in text:
                results.append(("error", "Model returned invalid classification scores."))
                continue
            names = list(labels)
            if self.score_override is not None:
                probs = {name: self.score_override[name] for name in names}
            else:
                prob = 1.0 / len(names)
                probs = {name: prob for name in names}
            choice = max(probs, key=probs.__getitem__)
            if self.return_tokens:
                token_count = 10 + (len(text.split()) if isinstance(text, str) else 0)
                results.append(("ok", (choice, probs, token_count)))
            else:
                results.append(("ok", (choice, probs)))
        return results

    def classify(self, text: str, labels, task: str = "decision", prompt: str | None = None):
        status, payload = self.classify_batch([(text, labels, task, prompt)])[0]
        if status == "ok":
            return payload[0], payload[1]
        if status == "input_too_long":
            raise WorkerInputTooLongError(payload)
        raise RuntimeError(payload)

    def close(self) -> None:
        self.loaded = False


class _DummyProc:
    def __init__(self, alive: bool = True) -> None:
        self.pid = 99999
        self._alive = alive
        self.joined = False
        self.killed = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.joined = True
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def close(self) -> None:
        pass


class _DummyConn:
    def __init__(self) -> None:
        self.sent: list[object] = []
        self.closed = False

    def send(self, value: object) -> None:
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class AdminConfigAndBindTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        for env in ({}, {"ADMIN_API_TOKEN": ""}, {"ADMIN_API_TOKEN": "   "}):
            cfg = load_admin_config(env)
            self.assertFalse(cfg.enabled)
            self.assertEqual(cfg.port, 9090)

        runtime = ChoiceRuntime()
        servers = build_servers(runtime=runtime, admin_config=load_admin_config({}))
        self.assertEqual(len(servers), 1)

    def test_rejects_weak_placeholder_or_malformed_tokens(self) -> None:
        bad_tokens = [
            "short",
            "15-bytes-exact!",
            "replace-with-a-random-admin-api-token",
            "REPLACE-WITH-UPPERCASE-TOKEN-1234",
            "changeme",
            "change-me",
            "has space inside token 123456",
            "has\nnewline-inside-token-123456",
            "non-ascii-token-café-123456",
            "non-ascii-nbsp-\u00a0-token-12345",
        ]
        for token in bad_tokens:
            with self.subTest(token=token):
                with self.assertRaises(ValueError):
                    load_admin_config({"ADMIN_API_TOKEN": token})

        exact_16 = load_admin_config({"ADMIN_API_TOKEN": "0123456789abcdef"})
        self.assertTrue(exact_16.enabled)
        self.assertNotIn("0123456789abcdef", repr(exact_16))
        self.assertNotIn("0123456789abcdef", str(exact_16))

    def test_port_and_cidr_validation(self) -> None:
        for bad_port in ("0", "-1", "65536", "8000", "abc", "+9090", "9_090"):
            with self.subTest(port=bad_port):
                with self.assertRaises(ValueError):
                    load_admin_config({"ADMIN_PORT": bad_port})

        cfg = load_admin_config({"ADMIN_PORT": "9191", "ADMIN_BIND_CIDR": "172.31.0.5/24"})
        self.assertEqual(cfg.port, 9191)
        self.assertEqual(cfg.bind_cidr, ipaddress.ip_network("172.31.0.0/24"))
        self.assertEqual(cfg.allow_cidr, ipaddress.ip_network("172.31.0.0/24"))

        cfg_explicit_allow = load_admin_config(
            {
                "ADMIN_BIND_CIDR": "172.31.0.0/24",
                "ADMIN_ALLOW_CIDR": "172.31.0.2/32",
            }
        )
        self.assertEqual(cfg_explicit_allow.allow_cidr, ipaddress.ip_network("172.31.0.2/32"))

        for bad_cidr in (
            "0.0.0.0/0",
            "::/0",
            "0.0.0.0/8",
            "::ffff:172.31.0.0/120",
            "172.31.0.2",
            "not-a-cidr",
        ):
            with self.subTest(cidr=bad_cidr):
                with self.assertRaises(ValueError):
                    load_admin_config({"ADMIN_BIND_CIDR": bad_cidr})
                with self.assertRaises(ValueError):
                    load_admin_config({"ADMIN_ALLOW_CIDR": bad_cidr})

    def test_main_validates_database_config_before_opening_admin_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "DB_BROKER_URL": "",
                    "DATABASE_URL": "",
                    "ADMIN_API_TOKEN": VALID_TOKEN,
                    "MODEL_PATH": tmpdir,
                },
                clear=False,
            ):
                with patch("app.__main__.logging.basicConfig"), patch("app.__main__.build_servers") as mock_build:
                    with self.assertRaises(SystemExit) as cm:
                        app_main()
                    self.assertEqual(cm.exception.code, 1)
                    mock_build.assert_not_called()

        with self.assertRaises(DatabaseConfigError):
            resolve_database_config({})

    def test_resolve_bind_addr(self) -> None:
        self.assertEqual(resolve_bind_addr(None, 9090), ("127.0.0.1", 9090))

        cidr = ipaddress.ip_network("172.31.0.0/24")
        resolved = resolve_bind_addr(
            cidr,
            9090,
            interface_ips=[
                ipaddress.ip_address("0.0.0.0"),
                ipaddress.ip_address("127.0.0.1"),
                ipaddress.ip_address("172.31.0.14"),
            ],
        )
        self.assertEqual(resolved, ("172.31.0.14", 9090))

        with self.assertRaises(RuntimeError):
            resolve_bind_addr(
                cidr,
                9090,
                interface_ips=[ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("10.0.0.5")],
            )

        local_ips = list_interface_ips()
        self.assertIn(ipaddress.ip_address("127.0.0.1"), local_ips)
        self.assertEqual(resolve_bind_addr(ipaddress.ip_network("127.0.0.0/8"), 9090), ("127.0.0.1", 9090))


class AdminAPISecurityAndContractTests(unittest.IsolatedAsyncioTestCase):
    def _make_client(
        self,
        *,
        peer: tuple[str, int] | None = ("127.0.0.1", 44444),
        allow_cidr: str | None = None,
        runtime: ChoiceRuntime | None = None,
    ) -> tuple[httpx.AsyncClient, ChoiceRuntime]:
        rt = runtime or ChoiceRuntime()
        env = {"ADMIN_API_TOKEN": VALID_TOKEN}
        if allow_cidr is not None:
            env["ADMIN_ALLOW_CIDR"] = allow_cidr
        cfg = load_admin_config(env)
        app = create_admin_app(rt, cfg)
        transport = httpx.ASGITransport(app=app, client=peer)  # type: ignore[arg-type]
        return httpx.AsyncClient(transport=transport, base_url="http://admin.local"), rt

    async def test_peer_cidr_and_xff_ignored(self) -> None:
        # Unset ADMIN_ALLOW_CIDR -> loopback only, even with valid token & spoofed XFF.
        async with self._make_client(peer=("172.31.0.2", 50000))[0] as cli:
            r = await cli.get(
                "/admin/v1/manifest",
                headers={"Authorization": f"Bearer {VALID_TOKEN}", "X-Forwarded-For": "127.0.0.1"},
            )
            self.assertEqual(r.status_code, 403)
            self.assertEqual(r.headers.get("cache-control"), "no-store")
            self.assertEqual(r.headers.get("x-content-type-options"), "nosniff")

        # Invalid/missing peer address -> 403.
        async with self._make_client(peer=("testclient", 50000))[0] as cli:
            r = await cli.get("/admin/v1/manifest", headers={"Authorization": f"Bearer {VALID_TOKEN}"})
            self.assertEqual(r.status_code, 403)

        # Explicit ADMIN_ALLOW_CIDR=172.31.0.2/32 rejects sibling container IP even with valid token & XFF.
        async with self._make_client(peer=("172.31.0.130", 50000), allow_cidr="172.31.0.2/32")[0] as cli:
            r = await cli.get(
                "/admin/v1/manifest",
                headers={"Authorization": f"Bearer {VALID_TOKEN}", "X-Forwarded-For": "172.31.0.2"},
            )
            self.assertEqual(r.status_code, 403)

        # IPv4-mapped IPv6 peer ::ffff:172.31.0.2 matches 172.31.0.2/32 and ignores XFF=8.8.8.8.
        async with self._make_client(peer=("::ffff:172.31.0.2", 50000), allow_cidr="172.31.0.2/32")[0] as cli:
            r = await cli.get(
                "/admin/v1/manifest",
                headers={"Authorization": f"Bearer {VALID_TOKEN}", "X-Forwarded-For": "8.8.8.8"},
            )
            self.assertEqual(r.status_code, 200)

        # IPv6 loopback ::1 is allowed when ADMIN_ALLOW_CIDR is unset, and Forwarded header is ignored.
        async with self._make_client(peer=("::1", 50000))[0] as cli:
            r = await cli.get(
                "/admin/v1/settings",
                headers={
                    "Authorization": f"Bearer {VALID_TOKEN}",
                    "Forwarded": "for=8.8.8.8",
                },
            )
            self.assertEqual(r.status_code, 200)

        async with self._make_client(peer=("172.31.0.50", 50000))[0] as cli:
            r = await cli.get(
                "/admin/v1/settings",
                headers={
                    "Authorization": f"Bearer {VALID_TOKEN}",
                    "Forwarded": "for=127.0.0.1",
                },
            )
            self.assertEqual(r.status_code, 403)

    async def test_bearer_auth_and_security_headers(self) -> None:
        async with self._make_client()[0] as cli:
            for headers in (
                {},
                {"Authorization": "Basic abcdef"},
                {"Authorization": "Bearer wrong-token"},
                {"Authorization": f"Bearer {VALID_TOKEN} extra"},
            ):
                r = await cli.get("/admin/v1/manifest", headers=headers)
                self.assertEqual(r.status_code, 401)
                self.assertIn("Bearer", r.headers.get("www-authenticate", ""))
                self.assertEqual(r.headers.get("cache-control"), "no-store")
                self.assertEqual(r.headers.get("x-content-type-options"), "nosniff")
                self.assertIn("error", r.json())

            # Unknown route and wrong method still return JSON error + required headers.
            r404 = await cli.get("/v1/classify", headers={"Authorization": f"Bearer {VALID_TOKEN}"})
            self.assertEqual(r404.status_code, 404)
            self.assertEqual(r404.headers.get("cache-control"), "no-store")
            self.assertEqual(r404.headers.get("x-content-type-options"), "nosniff")
            self.assertIn("error", r404.json())

            # Removed legacy /admin/v1/classify/batch returns 404.
            r404_legacy = await cli.post(
                "/admin/v1/classify/batch",
                headers={"Authorization": f"Bearer {VALID_TOKEN}"},
                json={"labels": ["a", "b"], "items": [{"id": "1", "text": "ok"}]},
            )
            self.assertEqual(r404_legacy.status_code, 404)

            r405 = await cli.post("/admin/v1/settings", headers={"Authorization": f"Bearer {VALID_TOKEN}"})
            self.assertEqual(r405.status_code, 405)
            self.assertEqual(r405.headers.get("cache-control"), "no-store")
            self.assertEqual(r405.headers.get("x-content-type-options"), "nosniff")
            self.assertIn("GET", r405.headers.get("allow", ""))

            # HEAD requests on authorized routes (FastAPI GET-only -> 405 with Allow: GET) and 401 paths have empty body.
            head_405 = await cli.head("/admin/v1/manifest", headers={"Authorization": f"Bearer {VALID_TOKEN}"})
            self.assertEqual(head_405.status_code, 405)
            self.assertEqual(head_405.content, b"")
            self.assertIn("GET", head_405.headers.get("allow", ""))
            self.assertEqual(head_405.headers.get("cache-control"), "no-store")
            self.assertEqual(head_405.headers.get("x-content-type-options"), "nosniff")

            head_401 = await cli.head("/admin/v1/settings")
            self.assertEqual(head_401.status_code, 401)
            self.assertEqual(head_401.content, b"")
            self.assertIn("Bearer", head_401.headers.get("www-authenticate", ""))

    async def test_raw_asgi_duplicate_auth_non_ascii_and_websocket_close(self) -> None:
        cfg = load_admin_config({"ADMIN_API_TOKEN": VALID_TOKEN})
        app = create_admin_app(ChoiceRuntime(), cfg)

        async def call_http(headers: list[tuple[bytes, bytes]], method: str = "GET", client=("127.0.0.1", 50000)):
            sent: list[dict[str, object]] = []

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(msg):
                sent.append(msg)

            await app(
                {
                    "type": "http",
                    "method": method,
                    "path": "/admin/v1/manifest",
                    "headers": headers,
                    "client": client,
                },
                receive,
                send,
            )
            return sent

        # Duplicate Authorization headers -> 401.
        dup_msgs = await call_http(
            [
                (b"authorization", f"Bearer {VALID_TOKEN}".encode("ascii")),
                (b"authorization", f"Bearer {VALID_TOKEN}".encode("ascii")),
            ]
        )
        self.assertEqual(dup_msgs[0]["status"], 401)

        # Non-ASCII raw bytes (including 0x85 NEL and 0xA0 NBSP) -> 401 cleanly without error.
        non_ascii_msgs = await call_http(
            [(b"authorization", b"Bearer \xc3\x85\xc3\xa0-invalid-token-123456")]
        )
        self.assertEqual(non_ascii_msgs[0]["status"], 401)

        # HEAD error response sends empty body matching _send_json_error HEAD branch.
        head_err_msgs = await call_http([], method="HEAD")
        self.assertEqual(head_err_msgs[0]["status"], 401)
        self.assertEqual(head_err_msgs[1]["body"], b"")

        # WebSocket scope -> closed immediately.
        ws_sent: list[dict[str, object]] = []

        async def ws_receive():
            return {"type": "websocket.connect"}

        async def ws_send(msg):
            ws_sent.append(msg)

        await app({"type": "websocket", "path": "/admin/v1/manifest", "headers": []}, ws_receive, ws_send)
        self.assertEqual(ws_sent, [{"type": "websocket.close", "code": 1008}])

    async def test_manifest_and_settings_contract_no_secrets(self) -> None:
        async with self._make_client()[0] as cli:
            headers = {"Authorization": f"bearer {VALID_TOKEN}"}
            man_resp = await cli.get("/admin/v1/manifest", headers=headers)
            self.assertEqual(man_resp.status_code, 200)
            self.assertEqual(man_resp.headers.get("cache-control"), "no-store")
            self.assertEqual(man_resp.headers.get("x-content-type-options"), "nosniff")
            man = man_resp.json()
            self.assertEqual(man["service"], "choice")
            self.assertEqual(man["version"], APP_VERSION)
            self.assertEqual(man["model"], MODEL_ID)
            self.assertEqual(man["models"], list(MODEL_IDS))
            self.assertEqual(man["capabilities"], ["settings", "classify", "api_keys"])
            self.assertEqual(
                man["endpoints"],
                {
                    "settings_get": "GET /admin/v1/settings",
                    "settings_update": "PATCH /admin/v1/settings",
                    "classify": "POST /admin/v1/classify",
                    "api_keys_list": "GET /admin/v1/api-keys",
                    "api_keys_create": "POST /admin/v1/api-keys",
                    "api_keys_revoke": "POST /admin/v1/api-keys/{id}/revoke",
                    "api_keys_delete": "DELETE /admin/v1/api-keys/{id}",
                },
            )
            self.assertEqual(man["settings_schema"], list(SETTINGS_SCHEMA))

            get_resp = await cli.get("/admin/v1/settings", headers=headers)
            self.assertEqual(get_resp.status_code, 200)
            settings = get_resp.json()
            self.assertEqual(
                settings,
                {
                    "accept_requests": True,
                    "idle_timeout_seconds": 180,
                    "max_text_chars": 8000,
                    "max_labels": 16,
                },
            )
            serialized = json.dumps(settings)
            self.assertNotIn("API_KEYS", serialized)
            self.assertNotIn(VALID_TOKEN, serialized)

    async def test_patch_validation_atomicity_and_body_limit(self) -> None:
        cli, rt = self._make_client()
        headers = {"Authorization": f"Bearer {VALID_TOKEN}", "Content-Type": "application/json"}
        async with cli:
            # Valid partial patch
            r = await cli.patch("/admin/v1/settings", headers=headers, json={"max_labels": 6})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["max_labels"], 6)

            # Empty object is valid no-op
            r_empty = await cli.patch("/admin/v1/settings", headers=headers, json={})
            self.assertEqual(r_empty.status_code, 200)
            self.assertEqual(r_empty.json()["max_labels"], 6)

            # Raise max_text_chars up to 16000 is valid
            r_max = await cli.patch("/admin/v1/settings", headers=headers, json={"max_text_chars": 16000})
            self.assertEqual(r_max.status_code, 200)
            self.assertEqual(r_max.json()["max_text_chars"], 16000)

            # Reject bool-as-int, float-as-int, null, strings, unknown keys, out-of-range, non-objects, malformed JSON
            bad_payloads = [
                '{"unknown_field": 1}',
                '{"idle_timeout_seconds": true}',
                '{"idle_timeout_seconds": false}',
                '{"max_text_chars": true}',
                '{"max_labels": false}',
                '{"idle_timeout_seconds": 12.5}',
                '{"idle_timeout_seconds": 12.0}',
                '{"idle_timeout_seconds": "180"}',
                '{"idle_timeout_seconds": null}',
                '{"accept_requests": 1}',
                '{"accept_requests": "true"}',
                '{"accept_requests": null}',
                '{"idle_timeout_seconds": -1}',
                '{"idle_timeout_seconds": 86401}',
                '{"max_text_chars": 99}',
                '{"max_text_chars": 16001}',
                '{"max_labels": 1}',
                '{"max_labels": 17}',
                '{"accept_requests": false, "max_labels": 99}',  # must remain atomic
                "[]",
                "123",
                "null",
                "",
                '{"accept_requests": false} {"max_labels": 4}',  # trailing JSON
                '{"idle_timeout_seconds": NaN}',
                '{"max_labels": 4, "max_labels": 5}',  # duplicate keys
                '{"accept_requests":' + ("[" * 2000) + ("]" * 2000) + "}",  # RecursionError guard
                "{not-json",
            ]
            for raw_body in bad_payloads:
                with self.subTest(body=raw_body):
                    bad = await cli.patch("/admin/v1/settings", headers=headers, content=raw_body.encode("utf-8"))
                    self.assertEqual(bad.status_code, 400)
                    self.assertIn("error", bad.json())
                    self.assertEqual(bad.headers.get("cache-control"), "no-store")
                    self.assertEqual(bad.headers.get("x-content-type-options"), "nosniff")

            # Verify atomicity: failed multi-field patch did NOT flip accept_requests or max_labels.
            snap = rt.snapshot()
            self.assertTrue(snap.accept_requests)
            self.assertEqual(snap.max_labels, 6)

            # Body > 16 KiB -> 413 (with Content-Length and via streaming generator without Content-Length).
            oversized = b'{"accept_requests":true,"pad":"' + (b"a" * MAX_ADMIN_BODY_BYTES) + b'"}'
            r413 = await cli.patch("/admin/v1/settings", headers=headers, content=oversized)
            self.assertEqual(r413.status_code, 413)
            self.assertEqual(r413.headers.get("cache-control"), "no-store")
            self.assertEqual(r413.headers.get("x-content-type-options"), "nosniff")

            async def chunked_oversized():
                yield b'{"accept_requests":true,"pad":"'
                yield b"a" * MAX_ADMIN_BODY_BYTES
                yield b'"}'

            r413_stream = await cli.patch("/admin/v1/settings", headers=headers, content=chunked_oversized())
            self.assertEqual(r413_stream.status_code, 413)

    async def test_admin_classify_contract_validation_and_guards(self) -> None:
        worker = _FakeWorker()
        rt = ChoiceRuntime()
        cli, _ = self._make_client(runtime=rt)
        headers = {"Authorization": f"Bearer {VALID_TOKEN}", "Content-Type": "application/json"}

        async with cli:
            # 1. Unauthenticated / outside CIDR -> 401 / 403
            r401 = await cli.post("/admin/v1/classify", json={"labels": ["a", "b"], "input": ["ok"]})
            self.assertEqual(r401.status_code, 401)

            async with self._make_client(peer=("172.31.0.99", 50000), allow_cidr="172.31.0.2/32", runtime=rt)[0] as bad_cli:
                r403 = await bad_cli.post(
                    "/admin/v1/classify",
                    headers={"Authorization": f"Bearer {VALID_TOKEN}", "X-Forwarded-For": "172.31.0.2"},
                    json={"labels": ["a", "b"], "input": ["ok"]},
                )
                self.assertEqual(r403.status_code, 403)

            # 2. Worker not yet attached -> 503
            r503 = await cli.post(
                "/admin/v1/classify",
                headers=headers,
                json={"labels": ["normal", "suspicious"], "input": ["GET /healthz 200"]},
            )
            self.assertEqual(r503.status_code, 503)
            self.assertEqual(r503.json(), {"error": "model worker unavailable"})
            self.assertEqual(r503.headers.get("cache-control"), "no-store")
            self.assertEqual(r503.headers.get("x-content-type-options"), "nosniff")

            # Attach worker and test happy path, descending sort, partial item errors, and token usage
            rt.attach_worker(worker)
            worker.score_override = {"normal": 0.15, "suspicious": 0.85}
            ok_resp = await cli.post(
                "/admin/v1/classify",
                headers=headers,
                json={
                    "model": "MLukauskas/amber-150m",
                    "prompt": "Classify the request log severity.",
                    "task": "log_triage",
                    "labels": {"normal": "Expected traffic", "suspicious": "Potential probe or abuse"},
                    "input": [
                        "GET /healthz 200",
                        "__TOO_LONG__ payload",
                        {"text": "__ERROR__ payload"},
                        {"text": {"path": "/etc/passwd", "status": 404}},
                    ],
                },
            )
            worker.score_override = None
            self.assertEqual(ok_resp.status_code, 200)
            self.assertEqual(ok_resp.headers.get("cache-control"), "no-store")
            self.assertEqual(ok_resp.headers.get("x-content-type-options"), "nosniff")
            self.assertEqual(
                ok_resp.json(),
                {
                    "object": "list",
                    "model": "amber-150m",
                    "data": [
                        {
                            "object": "classification",
                            "index": 0,
                            "prediction": "suspicious",
                            "score": 0.85,
                            "predictions": [
                                {"label": "suspicious", "score": 0.85},
                                {"label": "normal", "score": 0.15},
                            ],
                        },
                        {
                            "object": "classification",
                            "index": 1,
                            "error": "Prepared input length (8100 tokens) exceeds the 7,999-token limit.",
                        },
                        {
                            "object": "classification",
                            "index": 2,
                            "error": "Model returned invalid classification scores.",
                        },
                        {
                            "object": "classification",
                            "index": 3,
                            "prediction": "suspicious",
                            "score": 0.85,
                            "predictions": [
                                {"label": "suspicious", "score": 0.85},
                                {"label": "normal", "score": 0.15},
                            ],
                        },
                    ],
                    # Tokens: item 0 (10+3=13) + item 3 (10+1=11) = 24
                    "usage": {"prompt_tokens": 24, "total_tokens": 24},
                },
            )
            self.assertEqual(len(worker.batch_calls), 1)
            self.assertEqual(len(worker.batch_calls[0]), 4)

            # 2b. Single string and single {"text": ...} inputs also work
            single_resp = await cli.post(
                "/admin/v1/classify",
                headers=headers,
                json={"labels": ["a", "b"], "input": "single text"},
            )
            self.assertEqual(single_resp.status_code, 200)
            self.assertEqual(single_resp.json()["data"][0]["index"], 0)
            self.assertEqual(single_resp.json()["data"][0]["prediction"], "a")

            single_obj_resp = await cli.post(
                "/admin/v1/classify",
                headers=headers,
                json={"labels": ["a", "b"], "input": {"text": ["x", "y"]}},
            )
            self.assertEqual(single_obj_resp.status_code, 200)
            self.assertEqual(worker.calls[-1]["text"], '["x","y"]')

            # 3. Works even when accept_requests=false (maintenance mode)
            rt.apply_patch({"accept_requests": False})
            maint_resp = await cli.post(
                "/admin/v1/classify",
                headers=headers,
                json={
                    "labels": ["a", "b"],
                    "input": ["maintenance triage"],
                },
            )
            self.assertEqual(maint_resp.status_code, 200)
            self.assertEqual(maint_resp.json()["data"][0]["prediction"], "a")

            # 4. Worker runtime failure -> 503
            worker.fail_next_batch_exc = RuntimeError("worker crashed")
            crash_resp = await cli.post(
                "/admin/v1/classify",
                headers=headers,
                json={"labels": ["a", "b"], "input": ["hello"]},
            )
            self.assertEqual(crash_resp.status_code, 503)
            self.assertEqual(crash_resp.json(), {"error": "model worker unavailable"})

            # 5. Max batch of 64 items and 240-char descriptions + 512-char prompt succeed
            max_inputs = ["x" * 1000 for _ in range(64)]
            batch64 = await cli.post(
                "/admin/v1/classify",
                headers=headers,
                json={
                    "model": "amber-150m",
                    "prompt": "p" * 512,
                    "task": "t" * 64,
                    "labels": {"a": "d" * 240, "b": None},
                    "input": max_inputs,
                },
            )
            self.assertEqual(batch64.status_code, 200)
            self.assertEqual(len(batch64.json()["data"]), 64)

            # 5b. Explicit prompt: null is valid
            null_prompt = await cli.post(
                "/admin/v1/classify",
                headers=headers,
                json={"prompt": None, "labels": ["a", "b"], "input": ["hello"]},
            )
            self.assertEqual(null_prompt.status_code, 200)

            # 5c. Multilingual bge-m3-zeroshot-v2.0 canonical ID & alias work and return canonical model ID
            seen_admin_models: list[str] = []
            orig_cb = worker.classify_batch

            def _tracking_cb(items, *, model_id: str = MODEL_ID, **_kwargs):
                seen_admin_models.append(model_id)
                return orig_cb(items)

            with patch.object(worker, "classify_batch", side_effect=_tracking_cb):
                for req_model in (BGE_M3_MODEL_ID, f"MoritzLaurer/{BGE_M3_MODEL_ID}"):
                    r_bge = await cli.post(
                        "/admin/v1/classify",
                        headers=headers,
                        json={
                            "model": req_model,
                            "labels": ["정상", "이상"],
                            "input": ["한국어 로그 분류 테스트"],
                        },
                    )
                    self.assertEqual(r_bge.status_code, 200)
                    self.assertEqual(r_bge.json()["model"], BGE_M3_MODEL_ID)
            self.assertEqual(seen_admin_models, [BGE_M3_MODEL_ID, BGE_M3_MODEL_ID])

            # 6. Strict payload validation -> 400
            valid_base = {"labels": ["a", "b"], "input": ["ok"]}
            bad_cases: list[str] = [
                json.dumps({**valid_base, "extra": 1}),
                json.dumps({**valid_base, "model": "gpt-4o"}),
                json.dumps({**valid_base, "model": ""}),
                json.dumps({**valid_base, "model": None}),
                json.dumps({**valid_base, "model": 123}),
                json.dumps({"input": ["ok"]}),
                json.dumps({"labels": ["a", "b"]}),
                json.dumps({"labels": ["a", "b"], "items": [{"id": "0", "text": "legacy"}]}),
                json.dumps({**valid_base, "prompt": ""}),
                json.dumps({**valid_base, "prompt": "   "}),
                json.dumps({**valid_base, "prompt": "p" * 513}),
                json.dumps({**valid_base, "prompt": 123}),
                json.dumps({**valid_base, "task": ""}),
                json.dumps({**valid_base, "task": "   "}),
                json.dumps({**valid_base, "task": None}),
                json.dumps({**valid_base, "task": "t" * 65}),
                json.dumps({**valid_base, "labels": ["only_one"]}),
                json.dumps({**valid_base, "labels": [f"l{i}" for i in range(17)]}),
                json.dumps({**valid_base, "labels": ["dup", "dup"]}),
                json.dumps({**valid_base, "labels": ["", "b"]}),
                json.dumps({**valid_base, "labels": [" ", "b"]}),
                json.dumps({**valid_base, "labels": ["x" * 65, "b"]}),
                json.dumps({**valid_base, "labels": {"a": "", "b": "ok"}}),
                json.dumps({**valid_base, "labels": {"a": "   ", "b": "ok"}}),
                json.dumps({**valid_base, "labels": {"a": "d" * 241, "b": "ok"}}),
                json.dumps({**valid_base, "labels": {"a": 123, "b": "ok"}}),
                json.dumps({**valid_base, "labels": "not-a-list"}),
                json.dumps({"labels": ["a", "b"], "input": []}),
                json.dumps({"labels": ["a", "b"], "input": ["x"] * 65}),
                json.dumps({"labels": ["a", "b"], "input": 123}),
                json.dumps({"labels": ["a", "b"], "input": None}),
                json.dumps({"labels": ["a", "b"], "input": ""}),
                json.dumps({"labels": ["a", "b"], "input": "   "}),
                json.dumps({"labels": ["a", "b"], "input": [{}]}),
                json.dumps({"labels": ["a", "b"], "input": [{"text": "x", "extra": 1}]}),
                json.dumps({"labels": ["a", "b"], "input": [{"text": ""}]}),
                json.dumps({"labels": ["a", "b"], "input": [{"text": "   "}]}),
                json.dumps({"labels": ["a", "b"], "input": [{"text": 123}]}),
                json.dumps({"labels": ["a", "b"], "input": [{"text": True}]}),
                json.dumps({"labels": ["a", "b"], "input": ["x" * 16001]}),
                '{"labels":["a","b"],"labels":["c","d"],"input":["ok"]}',
                '{"labels":["a","b"],"input":[{"text":"ok","text":"dup"}]}',
                '{"labels":["a","b"],"input":["ok"],"prompt":NaN}',
                '{"labels":["a","b"],"input":["ok"],"prompt":Infinity}',
                '{"labels":["a","b"],"input":[{"text":' + ("[" * 9000) + ("]" * 9000) + "}]}",
                "[]",
                "{bad",
            ]
            for raw in bad_cases:
                with self.subTest(raw=raw[:80]):
                    bad = await cli.post("/admin/v1/classify", headers=headers, content=raw.encode("utf-8"))
                    self.assertEqual(bad.status_code, 400)
                    self.assertIn("error", bad.json())

            # Invalid UTF-8 -> 400
            bad_utf8 = await cli.post("/admin/v1/classify", headers=headers, content=b'{"labels":["a","b"],"input":["\xff"]}')
            self.assertEqual(bad_utf8.status_code, 400)

            # 7. Body > 256 KiB -> 413 (both direct and streamed)
            oversized = b'{"labels":["a","b"],"input":["' + (b"a" * MAX_ADMIN_CLASSIFY_BODY_BYTES) + b'"]}'
            r413 = await cli.post("/admin/v1/classify", headers=headers, content=oversized)
            self.assertEqual(r413.status_code, 413)

            async def chunked_oversized():
                yield b'{"labels":["a","b"],"input":["'
                yield b"a" * MAX_ADMIN_CLASSIFY_BODY_BYTES
                yield b'"]}'

            r413_stream = await cli.post("/admin/v1/classify", headers=headers, content=chunked_oversized())
            self.assertEqual(r413_stream.status_code, 413)

            # 8. Malformed worker payloads (non-mapping probs, NaN, bool) isolated as per-item errors (no 500)
            with patch.object(
                worker,
                "classify_batch",
                return_value=[
                    ("ok", ("a", None, 10)),
                    ("ok", ("a", {"a": float("nan"), "b": 0.5}, 10)),
                    ("ok", ("a", {"a": True, "b": 0.5}, 10)),
                    ("ok", ("a", {"a": 0.8, "b": 0.2}, 15)),
                ],
            ):
                r_malformed = await cli.post(
                    "/admin/v1/classify",
                    headers=headers,
                    json={"labels": ["a", "b"], "input": ["x0", "x1", "x2", "x3"]},
                )
                self.assertEqual(r_malformed.status_code, 200)
                m_body = r_malformed.json()
                self.assertIn("error", m_body["data"][0])
                self.assertIn("error", m_body["data"][1])
                self.assertIn("error", m_body["data"][2])
                self.assertEqual(m_body["data"][3]["prediction"], "a")
                self.assertEqual(m_body["usage"], {"prompt_tokens": 15, "total_tokens": 15})

        # 9. Mid-body client disconnect in _read_bounded_json_object returns 499 instead of 500
        cfg = load_admin_config({"ADMIN_API_TOKEN": VALID_TOKEN})
        admin_asgi = create_admin_app(rt, cfg)
        sent: list[dict[str, object]] = []
        step = 0

        async def disc_receive():
            nonlocal step
            if step == 0:
                step = 1
                return {"type": "http.request", "body": b'{"labels":', "more_body": True}
            return {"type": "http.disconnect"}

        async def disc_send(msg):
            sent.append(msg)

        await admin_asgi(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "path": "/admin/v1/classify",
                "raw_path": b"/admin/v1/classify",
                "query_string": b"",
                "headers": [
                    (b"authorization", f"Bearer {VALID_TOKEN}".encode("ascii")),
                    (b"content-type", b"application/json"),
                ],
                "client": ("127.0.0.1", 51114),
                "server": ("127.0.0.1", 9090),
            },
            disc_receive,
            disc_send,
        )
        self.assertEqual(sent[0]["status"], 499)
        self.assertEqual(json.loads(sent[1]["body"].decode("utf-8")), {"error": "request cancelled"})


class LiveSettingsAndWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_settings_affect_public_classify_and_healthz(self) -> None:
        runtime = ChoiceRuntime()
        fake_worker = _FakeWorker(idle_timeout=180.0)
        admin_cfg = load_admin_config({"ADMIN_API_TOKEN": VALID_TOKEN})
        store, public_key = _make_store_with_key()

        with patch("app.api.ModelWorker", return_value=fake_worker):
            public_app = create_app(runtime=runtime, api_key_store=store)
            admin_app = create_admin_app(runtime=runtime, config=admin_cfg, api_key_store=store)

            async with public_app.router.lifespan_context(public_app):
                async with (
                    httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=public_app, client=("127.0.0.1", 50001)),
                        base_url="http://public.local",
                    ) as pub_cli,
                    httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=admin_app, client=("127.0.0.1", 50002)),
                        base_url="http://admin.local",
                    ) as adm_cli,
                ):
                    pub_headers = {"Authorization": f"Bearer {public_key}"}
                    adm_headers = {"Authorization": f"Bearer {VALID_TOKEN}"}

                    # Public app must NOT expose admin routes.
                    self.assertEqual((await pub_cli.get("/admin/v1/manifest")).status_code, 404)

                    # Normal classify works.
                    ok = await pub_cli.post(
                        "/v1/classify",
                        headers=pub_headers,
                        json={"input": "x" * 200, "labels": ["a", "b", "c", "d"]},
                    )
                    self.assertEqual(ok.status_code, 200)
                    self.assertEqual(ok.json()["data"][0]["prediction"], "a")

                    # Lower max_text_chars=120 and max_labels=3 via Admin API.
                    patch_limits = await adm_cli.patch(
                        "/admin/v1/settings",
                        headers=adm_headers,
                        json={"max_text_chars": 120, "max_labels": 3, "idle_timeout_seconds": 45},
                    )
                    self.assertEqual(patch_limits.status_code, 200)
                    self.assertEqual(fake_worker.idle_timeout, 45.0)

                    too_long = await pub_cli.post(
                        "/v1/classify",
                        headers=pub_headers,
                        json={"input": "x" * 121, "labels": ["a", "b"]},
                    )
                    self.assertEqual(too_long.status_code, 422)

                    too_many_labels = await pub_cli.post(
                        "/v1/classify",
                        headers=pub_headers,
                        json={"input": "x" * 100, "labels": ["a", "b", "c", "d"]},
                    )
                    self.assertEqual(too_many_labels.status_code, 422)

                    within_limits = await pub_cli.post(
                        "/v1/classify",
                        headers=pub_headers,
                        json={"input": "x" * 120, "labels": ["a", "b", "c"]},
                    )
                    self.assertEqual(within_limits.status_code, 200)

                    # Pause requests via accept_requests=false -> /v1/classify 503, /healthz & /v1/models 200.
                    paused = await adm_cli.patch(
                        "/admin/v1/settings",
                        headers=adm_headers,
                        json={"accept_requests": False},
                    )
                    self.assertEqual(paused.status_code, 200)

                    r503 = await pub_cli.post(
                        "/v1/classify",
                        headers=pub_headers,
                        json={"input": "ok", "labels": ["a", "b"]},
                    )
                    self.assertEqual(r503.status_code, 503)
                    self.assertIn("error", r503.json())

                    # Unauthenticated request still returns 401 even while paused.
                    r401 = await pub_cli.post("/v1/classify", json={"input": "ok", "labels": ["a", "b"]})
                    self.assertEqual(r401.status_code, 401)

                    # Healthz, /v1/models, and Admin API stay up.
                    hz = await pub_cli.get("/healthz")
                    self.assertEqual(hz.status_code, 200)
                    self.assertEqual(hz.json()["ok"], True)
                    mods = await pub_cli.get("/v1/models", headers=pub_headers)
                    self.assertEqual(mods.status_code, 200)

                    # Re-enable requests -> /v1/classify recovers immediately.
                    resumed = await adm_cli.patch(
                        "/admin/v1/settings",
                        headers=adm_headers,
                        json={"accept_requests": True},
                    )
                    self.assertEqual(resumed.status_code, 200)
                    r200 = await pub_cli.post(
                        "/v1/classify",
                        headers=pub_headers,
                        json={"input": "ok", "labels": ["a", "b"]},
                    )
                    self.assertEqual(r200.status_code, 200)

    def test_worker_set_idle_timeout_behaviors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            w = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0)
            proc = _DummyProc(alive=True)
            conn = _DummyConn()
            w._proc = proc  # type: ignore[assignment]
            w._conn = conn  # type: ignore[assignment]
            w._last_used = time.monotonic() - 10.0
            w._schedule_timer_locked()
            self.assertIsNotNone(w._timer)

            # 1. Change to 0 -> cancels timer and keeps worker resident.
            w.set_idle_timeout(0)
            self.assertIsNone(w._timer)
            self.assertTrue(w.is_loaded())

            # Stale timer generation callback must NOT unload worker when timeout is 0.
            w._unload_if_idle(timer_gen=1)
            self.assertTrue(w.is_loaded())

            # 2. Change from 0 to 15s (idle_for ~10s -> remaining ~5s) -> schedules timer without immediate unload.
            w.set_idle_timeout(15)
            self.assertIsNotNone(w._timer)
            self.assertTrue(w.is_loaded())

            # 2b. Raise timeout (15s -> 60s) -> reschedules timer cleanly and keeps worker loaded.
            w.set_idle_timeout(60)
            self.assertIsNotNone(w._timer)
            self.assertEqual(w.idle_timeout, 60.0)
            self.assertTrue(w.is_loaded())

            # 3. Lower below current idle time (5s < 10s idle_for) -> immediately evicts worker.
            w.set_idle_timeout(5)
            self.assertFalse(w.is_loaded())
            self.assertTrue(conn.closed)
            self.assertEqual(conn.sent, [None])
            self.assertTrue(proc.joined)

            # 4. Stale/exited _proc cleaned up cleanly on set_idle_timeout, and closed proc does not crash is_loaded().
            dead_proc = _DummyProc(alive=False)
            dead_conn = _DummyConn()
            w._proc = dead_proc  # type: ignore[assignment]
            w._conn = dead_conn  # type: ignore[assignment]
            w.set_idle_timeout(30)
            self.assertIsNone(w._proc)
            self.assertTrue(dead_conn.closed)
            self.assertFalse(w.is_loaded())

            class _ClosedProc:
                def is_alive(self) -> bool:
                    raise ValueError("process object is closed")

            w._proc = _ClosedProc()  # type: ignore[assignment]
            self.assertFalse(w.is_loaded())

    async def test_admin_control_plane_not_starved_by_public_load_or_timeout_block(self) -> None:
        runtime = ChoiceRuntime()
        entered_timeout = threading.Event()
        release_timeout = threading.Event()
        entered_public = threading.Event()
        release_public = threading.Event()

        class _SlowWorker(_FakeWorker):
            def set_idle_timeout(self, idle_timeout: float) -> None:
                entered_timeout.set()
                release_timeout.wait(timeout=5.0)
                super().set_idle_timeout(idle_timeout)

            def classify_batch(self, items, *, plane: str = "interactive", **_kwargs):
                if plane == "interactive":
                    entered_public.set()
                    release_public.wait(timeout=5.0)
                return super().classify_batch(items)

        slow_worker = _SlowWorker()
        admin_cfg = load_admin_config({"ADMIN_API_TOKEN": VALID_TOKEN})
        store, public_key = _make_store_with_key()

        with patch("app.api.ModelWorker", return_value=slow_worker):
            public_app = create_app(runtime=runtime, api_key_store=store)
            admin_app = create_admin_app(runtime=runtime, config=admin_cfg, api_key_store=store)

            async with public_app.router.lifespan_context(public_app):
                async with (
                    httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=public_app, client=("127.0.0.1", 50001)),
                        base_url="http://public.local",
                    ) as pub_cli,
                    httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=admin_app, client=("127.0.0.1", 50002)),
                        base_url="http://admin.local",
                    ) as adm_cli,
                ):
                    pub_headers = {"Authorization": f"Bearer {public_key}"}
                    adm_headers = {"Authorization": f"Bearer {VALID_TOKEN}"}

                    # Saturate all 24 public classify slots + block an idle_timeout patch.
                    pub_tasks = [
                        asyncio.create_task(
                            pub_cli.post(
                                "/v1/classify",
                                headers=pub_headers,
                                json={"input": f"slow-{i}", "labels": ["a", "b"]},
                            )
                        )
                        for i in range(25)
                    ]
                    while not entered_public.is_set():
                        await asyncio.sleep(0.01)

                    idle_patch_task = asyncio.create_task(
                        adm_cli.patch(
                            "/admin/v1/settings",
                            headers=adm_headers,
                            json={"idle_timeout_seconds": 30},
                        )
                    )
                    while not entered_timeout.is_set():
                        await asyncio.sleep(0.01)

                    # Admin accept_requests=false & /admin/v1/classify MUST respond immediately without hanging!
                    r_pause = await asyncio.wait_for(
                        adm_cli.patch(
                            "/admin/v1/settings",
                            headers=adm_headers,
                            json={"accept_requests": False},
                        ),
                        timeout=2.0,
                    )
                    self.assertEqual(r_pause.status_code, 200)
                    self.assertFalse(r_pause.json()["accept_requests"])

                    r_batch = await asyncio.wait_for(
                        adm_cli.post(
                            "/admin/v1/classify",
                            headers=adm_headers,
                            json={"labels": ["normal", "alert"], "input": ["triage"]},
                        ),
                        timeout=2.0,
                    )
                    self.assertEqual(r_batch.status_code, 200)

                    release_timeout.set()
                    release_public.set()
                    await asyncio.wait_for(idle_patch_task, timeout=2.0)
                    pub_resps = await asyncio.wait_for(asyncio.gather(*pub_tasks), timeout=5.0)
                    self.assertTrue(all(r.status_code in (200, 503) for r in pub_resps))

    async def test_admin_settings_limiter_isolated_from_classify_saturation(self) -> None:
        runtime = ChoiceRuntime()
        active_batches = 0
        active_lock = threading.Lock()
        saturated = threading.Event()
        release_batches = threading.Event()
        seen_planes: list[str] = []
        seen_cancel_events: list[threading.Event] = []

        class _SlowBatchWorker(_FakeWorker):
            def classify_batch(self, items, *, cancel_event: threading.Event | None = None, plane: str = "interactive"):
                nonlocal active_batches
                with active_lock:
                    seen_planes.append(plane)
                    if cancel_event is not None:
                        seen_cancel_events.append(cancel_event)
                    active_batches += 1
                    if active_batches >= 4:
                        saturated.set()
                release_batches.wait(timeout=5.0)
                return super().classify_batch(items)

        worker = _SlowBatchWorker()
        runtime.attach_worker(worker)
        cli, _ = self._make_client(runtime=runtime)
        headers = {"Authorization": f"Bearer {VALID_TOKEN}", "Content-Type": "application/json"}

        async with cli:
            # Launch 6 concurrent slow /admin/v1/classify calls to saturate classify_limiter (capacity=4)
            batch_tasks = [
                asyncio.create_task(
                    cli.post(
                        "/admin/v1/classify",
                        headers=headers,
                        json={"labels": ["normal", "alert"], "input": [f"heavy log {idx}"]},
                    )
                )
                for idx in range(6)
            ]
            for _ in range(200):
                if saturated.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(saturated.is_set(), "classify_limiter should be saturated by 4 active calls")
            self.assertTrue(all(p == "admin" for p in seen_planes))

            # Emergency PATCH /admin/v1/settings (accept_requests=false) MUST NOT be starved!
            r_pause = await asyncio.wait_for(
                cli.patch(
                    "/admin/v1/settings",
                    headers=headers,
                    json={"accept_requests": False},
                ),
                timeout=1.5,
            )
            self.assertEqual(r_pause.status_code, 200)
            self.assertFalse(r_pause.json()["accept_requests"])

            # Cancelling one in-flight admin batch task propagates cancel_event promptly
            batch_tasks[0].cancel()
            with self.assertRaises(asyncio.CancelledError):
                await batch_tasks[0]
            self.assertTrue(any(ev.is_set() for ev in seen_cancel_events))

            release_batches.set()
            remaining_resps = await asyncio.wait_for(asyncio.gather(*batch_tasks[1:]), timeout=3.0)
            self.assertTrue(all(r.status_code == 200 for r in remaining_resps))

    def _make_client(
        self,
        *,
        peer: tuple[str, int] | None = ("127.0.0.1", 44444),
        allow_cidr: str | None = None,
        runtime: ChoiceRuntime | None = None,
    ) -> tuple[httpx.AsyncClient, ChoiceRuntime]:
        rt = runtime or ChoiceRuntime()
        env = {"ADMIN_API_TOKEN": VALID_TOKEN}
        if allow_cidr is not None:
            env["ADMIN_ALLOW_CIDR"] = allow_cidr
        cfg = load_admin_config(env)
        app = create_admin_app(rt, cfg)
        transport = httpx.ASGITransport(app=app, client=peer)  # type: ignore[arg-type]
        return httpx.AsyncClient(transport=transport, base_url="http://admin.local"), rt


class LiveDualUvicornIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_dual_uvicorn_servers_and_xff_bypass_prevention(self) -> None:
        main_port = _pick_free_port()
        admin_port = _pick_free_port()
        while admin_port == main_port:
            admin_port = _pick_free_port()

        runtime = ChoiceRuntime()
        fake_worker = _FakeWorker(idle_timeout=180.0)
        store, public_key = _make_store_with_key()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "FORWARDED_ALLOW_IPS": "*",
                    "ADMIN_API_TOKEN": VALID_TOKEN,
                    "ADMIN_PORT": str(admin_port),
                    "MODEL_PATH": tmpdir,
                },
                clear=False,
            ):
                with patch("app.api.ModelWorker", return_value=fake_worker):
                    admin_cfg = load_admin_config()
                    servers = build_servers(
                        runtime=runtime,
                        admin_config=admin_cfg,
                        api_key_store=store,
                        main_host="127.0.0.1",
                        main_port=main_port,
                        admin_host="127.0.0.1",
                    )
                    self.assertEqual(len(servers), 2)
                    serve_task = asyncio.create_task(serve_servers(servers))
                    try:
                        # Wait until both listeners accept connections.
                        async with httpx.AsyncClient() as cli:
                            for _ in range(100):
                                try:
                                    r_pub = await cli.get(f"http://127.0.0.1:{main_port}/healthz")
                                    r_adm = await cli.get(
                                        f"http://127.0.0.1:{admin_port}/admin/v1/manifest",
                                        headers={"Authorization": f"Bearer {VALID_TOKEN}"},
                                    )
                                    if r_pub.status_code == 200 and r_adm.status_code == 200:
                                        break
                                except httpx.ConnectError:
                                    await asyncio.sleep(0.05)
                            else:
                                self.fail("Dual uvicorn servers did not become ready in time")

                            # Even with FORWARDED_ALLOW_IPS=* in env, admin server disables proxy_headers:
                            # sending X-Forwarded-For: 8.8.8.8 from 127.0.0.1 still uses TCP peer 127.0.0.1 (200).
                            r_xff = await cli.get(
                                f"http://127.0.0.1:{admin_port}/admin/v1/settings",
                                headers={
                                    "Authorization": f"Bearer {VALID_TOKEN}",
                                    "X-Forwarded-For": "8.8.8.8",
                                },
                            )
                            self.assertEqual(r_xff.status_code, 200)
                            self.assertEqual(r_xff.headers.get("cache-control"), "no-store")
                            self.assertEqual(r_xff.headers.get("x-content-type-options"), "nosniff")

                            # Mutate accept_requests=false on admin port, observe 503 on public port over real TCP!
                            r_patch = await cli.patch(
                                f"http://127.0.0.1:{admin_port}/admin/v1/settings",
                                headers={"Authorization": f"Bearer {VALID_TOKEN}"},
                                json={"accept_requests": False},
                            )
                            self.assertEqual(r_patch.status_code, 200)

                            r_cls = await cli.post(
                                f"http://127.0.0.1:{main_port}/v1/classify",
                                headers={"Authorization": f"Bearer {public_key}"},
                                json={"input": "hello world", "labels": ["a", "b"]},
                            )
                            self.assertEqual(r_cls.status_code, 503)

                            # Internal classify on :9090 still works during maintenance mode
                            # and shares the exact same fake_worker instance with :8000!
                            r_adm_batch = await cli.post(
                                f"http://127.0.0.1:{admin_port}/admin/v1/classify",
                                headers={"Authorization": f"Bearer {VALID_TOKEN}"},
                                json={
                                    "labels": ["normal", "alert"],
                                    "input": ["x" * 3500 for _ in range(30)],
                                },
                            )
                            self.assertEqual(r_adm_batch.status_code, 200)
                            self.assertEqual(r_adm_batch.json()["object"], "list")
                            self.assertEqual(r_adm_batch.json()["model"], "amber-150m")
                            self.assertEqual(len(r_adm_batch.json()["data"]), 30)
                            self.assertEqual(len(fake_worker.batch_calls), 1)
                            self.assertIs(runtime.get_worker(), fake_worker)
                    finally:
                        for srv in servers:
                            srv.should_exit = True
                        await serve_task

    async def test_serve_servers_external_cancellation_cleans_up_both_listeners(self) -> None:
        main_port = _pick_free_port()
        admin_port = _pick_free_port()
        while admin_port == main_port:
            admin_port = _pick_free_port()

        runtime = ChoiceRuntime()
        fake_worker = _FakeWorker(idle_timeout=180.0)
        store, _ = _make_store_with_key()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "ADMIN_API_TOKEN": VALID_TOKEN,
                    "ADMIN_PORT": str(admin_port),
                    "MODEL_PATH": tmpdir,
                },
                clear=False,
            ):
                with patch("app.api.ModelWorker", return_value=fake_worker):
                    servers = build_servers(
                        runtime=runtime,
                        admin_config=load_admin_config(),
                        api_key_store=store,
                        main_host="127.0.0.1",
                        main_port=main_port,
                        admin_host="127.0.0.1",
                    )
                    task = asyncio.create_task(serve_servers(servers))
                    async with httpx.AsyncClient() as cli:
                        for _ in range(100):
                            try:
                                r = await cli.get(
                                    f"http://127.0.0.1:{admin_port}/admin/v1/manifest",
                                    headers={"Authorization": f"Bearer {VALID_TOKEN}"},
                                )
                                if r.status_code == 200:
                                    break
                            except httpx.ConnectError:
                                await asyncio.sleep(0.05)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertTrue(all(srv.should_exit for srv in servers))


if __name__ == "__main__":
    unittest.main()
