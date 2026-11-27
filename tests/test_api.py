"""Contract, schema, concurrency, and boundary tests for the public Choice API (:8000)."""

from __future__ import annotations

import asyncio
import json
import threading
import unittest
from unittest.mock import patch

import httpx

from app.api import MAX_PUBLIC_BODY_BYTES, create_app
from app.api_keys import APIKeyStore, InMemoryAPIKeyBackend
from app.runtime import MODEL_ID, MODEL_OWNED_BY, ChoiceRuntime
from app.worker import WorkerCancelledError, WorkerInputTooLongError


def _make_store_with_key() -> tuple[APIKeyStore, str]:
    store = APIKeyStore(InMemoryAPIKeyBackend())
    plaintext, _record = store.create_key(name="test-client", owner="tests", created_by="suite")
    return store, plaintext


class _FakePublicWorker:
    def __init__(self, idle_timeout: float = 180.0, *, return_tokens: bool = True) -> None:
        self.idle_timeout = float(idle_timeout)
        self.loaded = False
        self.return_tokens = return_tokens
        self.calls: list[dict[str, object]] = []
        self.score_override: dict[str, float] | None = None

    def is_loaded(self) -> bool:
        return self.loaded

    def start(self) -> None:
        self.loaded = True

    def set_idle_timeout(self, idle_timeout: float) -> None:
        self.idle_timeout = float(idle_timeout)

    def classify_batch(self, items):
        self.loaded = True
        results = []
        for text, labels, task, prompt in items:
            self.calls.append({"text": text, "labels": labels, "task": task, "prompt": prompt})
            if "__OVER_TOKENS__" in text:
                results.append(("input_too_long", "Prepared input length (8050 tokens) exceeds the 7,999-token limit."))
                continue
            if "__ERROR__" in text:
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
                token_count = 10 + len(text.split())
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


def _assert_openai_error(tc: unittest.TestCase, resp: httpx.Response, expected_status: int, *, expected_code: str | None = None) -> dict:
    tc.assertEqual(resp.status_code, expected_status, resp.text)
    body = resp.json()
    tc.assertIn("error", body)
    err = body["error"]
    tc.assertIsInstance(err, dict)
    tc.assertIsInstance(err.get("message"), str)
    tc.assertTrue(err["message"])
    tc.assertIsInstance(err.get("type"), str)
    tc.assertTrue(err["type"])
    tc.assertIsInstance(err.get("code"), str)
    tc.assertTrue(err["code"])
    if expected_code is not None:
        tc.assertEqual(err["code"], expected_code)
    return err


class PublicAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_public_v1_classify_models_healthz_and_removed_legacy_routes(self) -> None:
        runtime = ChoiceRuntime()
        fake_worker = _FakePublicWorker()
        store, public_key = _make_store_with_key()
        headers = {"Authorization": f"Bearer {public_key}"}

        with patch("app.api.ModelWorker", return_value=fake_worker):
            app = create_app(runtime=runtime, api_key_store=store)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 50010)),
                    base_url="http://public.local",
                ) as cli:
                    # OpenAPI schema & Swagger UI work without PydanticUserError and expose only /v1 + /healthz
                    openapi_resp = await cli.get("/openapi.json")
                    self.assertEqual(openapi_resp.status_code, 200, openapi_resp.text)
                    schema = openapi_resp.json()
                    self.assertEqual(set(schema["paths"].keys()), {"/v1/classify", "/v1/models", "/healthz"})
                    sec_schemes = schema["components"]["securitySchemes"]
                    self.assertIn("BearerKey", sec_schemes)
                    self.assertIn("ApiKeyHeader", sec_schemes)
                    docs_resp = await cli.get("/docs")
                    self.assertEqual(docs_resp.status_code, 200)

                    # Legacy endpoints are completely removed (404)
                    r_old_cls = await cli.post(
                        "/api/classify",
                        headers=headers,
                        json={"text": "hello", "labels": ["a", "b"]},
                    )
                    self.assertEqual(r_old_cls.status_code, 404)
                    r_old_hz = await cli.get("/api/healthz")
                    self.assertEqual(r_old_hz.status_code, 404)

                    # /healthz (unauthenticated)
                    hz = await cli.get("/healthz")
                    self.assertEqual(hz.status_code, 200)
                    self.assertEqual(hz.json(), {"ok": True, "model_loaded": False, "model": MODEL_ID})

                    # GET /v1/models requires auth and returns OpenAI error envelope on 401
                    m401 = await cli.get("/v1/models")
                    _assert_openai_error(self, m401, 401, expected_code="invalid_api_key")
                    self.assertIn("Bearer", m401.headers.get("www-authenticate", ""))

                    m_bearer = await cli.get("/v1/models", headers=headers)
                    self.assertEqual(m_bearer.status_code, 200)
                    self.assertEqual(
                        m_bearer.json(),
                        {
                            "object": "list",
                            "data": [{"id": MODEL_ID, "object": "model", "owned_by": MODEL_OWNED_BY}],
                        },
                    )
                    m_xapi = await cli.get("/v1/models", headers={"X-API-Key": public_key})
                    self.assertEqual(m_xapi.status_code, 200)
                    self.assertEqual(m_xapi.json(), m_bearer.json())

                    # Missing auth on /v1/classify -> 401 OpenAI error BEFORE JSON decoding (even for malformed JSON / invalid UTF-8)
                    r401 = await cli.post("/v1/classify", json={"input": "hello", "labels": ["a", "b"]})
                    _assert_openai_error(self, r401, 401, expected_code="invalid_api_key")
                    self.assertIn("Bearer", r401.headers.get("www-authenticate", ""))
                    r401_empty = await cli.post("/v1/classify", json={})
                    _assert_openai_error(self, r401_empty, 401, expected_code="invalid_api_key")
                    for raw_bad in (b"{bad", b'{"input":"\xff"}', b'{"input":' + (b"[" * 2000) + (b"]" * 2000) + b"}"):
                        r401_raw = await cli.post(
                            "/v1/classify",
                            headers={"Content-Type": "application/json"},
                            content=raw_bad,
                        )
                        _assert_openai_error(self, r401_raw, 401, expected_code="invalid_api_key")
                        self.assertIn("Bearer", r401_raw.headers.get("www-authenticate", ""))

                    # Maintenance mode (accept_requests=False) returns 503 BEFORE JSON decoding on malformed JSON
                    runtime.apply_patch({"accept_requests": False})
                    r503_bad_json = await cli.post(
                        "/v1/classify",
                        headers={**headers, "Content-Type": "application/json"},
                        content=b"{bad",
                    )
                    _assert_openai_error(self, r503_bad_json, 503, expected_code="service_unavailable")
                    runtime.apply_patch({"accept_requests": True})

                    # X-API-Key header works identically to Authorization: Bearer
                    r_xapi = await cli.post(
                        "/v1/classify",
                        headers={"X-API-Key": public_key},
                        json={"input": "hello world", "labels": ["a", "b"]},
                    )
                    self.assertEqual(r_xapi.status_code, 200)
                    self.assertEqual(
                        r_xapi.json(),
                        {
                            "object": "list",
                            "model": MODEL_ID,
                            "data": [
                                {
                                    "object": "classification",
                                    "index": 0,
                                    "prediction": "a",
                                    "score": 0.5,
                                    "predictions": [
                                        {"label": "a", "score": 0.5},
                                        {"label": "b", "score": 0.5},
                                    ],
                                }
                            ],
                            "usage": {"prompt_tokens": 12, "total_tokens": 12},
                        },
                    )

                    # Predictions sorted descending by score; ties preserve requested label order
                    fake_worker.score_override = {"low": 0.1, "tie_first": 0.45, "tie_second": 0.45}
                    r_sorted = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={
                            "model": "MLukauskas/amber-150m",
                            "input": "I was charged twice.",
                            "task": "department",
                            "prompt": "Which team handles this?",
                            "labels": {
                                "low": "Low priority",
                                "tie_first": "First tie",
                                "tie_second": None,
                            },
                        },
                    )
                    fake_worker.score_override = None
                    self.assertEqual(r_sorted.status_code, 200)
                    sorted_body = r_sorted.json()
                    self.assertEqual(sorted_body["model"], MODEL_ID)
                    item0 = sorted_body["data"][0]
                    self.assertEqual(item0["prediction"], "tie_first")
                    self.assertEqual(item0["score"], 0.45)
                    self.assertEqual(
                        item0["predictions"],
                        [
                            {"label": "tie_first", "score": 0.45},
                            {"label": "tie_second", "score": 0.45},
                            {"label": "low", "score": 0.1},
                        ],
                    )
                    self.assertEqual(item0["predictions"][0]["label"], item0["prediction"])
                    self.assertEqual(item0["predictions"][0]["score"], item0["score"])

                    # Unknown model -> HTTP 400 OpenAI error
                    for bad_model in ("gpt-4o", "amber-999m", "", "   "):
                        r_bad_model = await cli.post(
                            "/v1/classify",
                            headers=headers,
                            json={"model": bad_model, "input": "hello", "labels": ["a", "b"]},
                        )
                        _assert_openai_error(self, r_bad_model, 400, expected_code="model_not_found")

                    # Healthz reflects model_loaded=True after request
                    hz_loaded = await cli.get("/healthz")
                    self.assertEqual(hz_loaded.json(), {"ok": True, "model_loaded": True, "model": MODEL_ID})

                    # Single {"text": dict}, single {"text": list}, and mixed batch list inputs
                    r_obj = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": {"text": {"cmd": "git status"}}, "labels": ["safe", "unsafe"]},
                    )
                    self.assertEqual(r_obj.status_code, 200)
                    self.assertEqual(fake_worker.calls[-1]["text"], '{"cmd":"git status"}')

                    r_arr = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": {"text": ["step1", "step2"]}, "labels": ["ok", "fail"]},
                    )
                    self.assertEqual(r_arr.status_code, 200)
                    self.assertEqual(fake_worker.calls[-1]["text"], '["step1","step2"]')

                    r_batch = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={
                            "model": "amber-150m",
                            "input": [
                                "first message",
                                {"text": "second message here"},
                                {"text": {"k": "v"}},
                                {"text": ["a", "b"]},
                            ],
                            "labels": ["billing", "technical"],
                        },
                    )
                    self.assertEqual(r_batch.status_code, 200)
                    batch_body = r_batch.json()
                    self.assertEqual(batch_body["object"], "list")
                    self.assertEqual(len(batch_body["data"]), 4)
                    self.assertEqual([d["index"] for d in batch_body["data"]], [0, 1, 2, 3])
                    # Tokens: (10+2) + (10+3) + (10+1) + (10+1) = 47
                    self.assertEqual(batch_body["usage"], {"prompt_tokens": 47, "total_tokens": 47})

                    # 64-item batch succeeds, including payloads between 128 KiB and 256 KiB
                    r_64 = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": ["x" * 2500 for _ in range(64)], "labels": ["a", "b"]},
                    )
                    self.assertEqual(r_64.status_code, 200)
                    self.assertEqual(len(r_64.json()["data"]), 64)

                    # Empty input list or > 64 items -> 422 OpenAI error
                    r_empty_list = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": [], "labels": ["a", "b"]},
                    )
                    _assert_openai_error(self, r_empty_list, 422, expected_code="validation_error")

                    r_65 = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": ["x"] * 65, "labels": ["a", "b"]},
                    )
                    _assert_openai_error(self, r_65, 422, expected_code="validation_error")

                    # Default runtime max_text_chars=8000 allows 8000 and rejects 8001
                    r_8000 = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": "x" * 8000, "labels": ["a", "b"]},
                    )
                    self.assertEqual(r_8000.status_code, 200)

                    r_8001 = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": ["ok", "x" * 8001], "labels": ["a", "b"]},
                    )
                    _assert_openai_error(self, r_8001, 422, expected_code="input_too_long")

                    # Live raise to max_text_chars=16000 allows 16000; schema rejects 16001
                    runtime.apply_patch({"max_text_chars": 16000})
                    r_16000 = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": "x" * 16000, "labels": ["a", "b"]},
                    )
                    self.assertEqual(r_16000.status_code, 200)

                    r_16001 = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": "x" * 16001, "labels": ["a", "b"]},
                    )
                    _assert_openai_error(self, r_16001, 422, expected_code="validation_error")

                    # Token-budget error from worker (single and inside batch) -> 422 OpenAI error
                    r_tok = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": "__OVER_TOKENS__", "labels": ["a", "b"]},
                    )
                    err_tok = _assert_openai_error(self, r_tok, 422, expected_code="context_length_exceeded")
                    self.assertIn("7,999", err_tok["message"])

                    r_tok_batch = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": ["ok", "__OVER_TOKENS__"], "labels": ["a", "b"]},
                    )
                    err_tok_b = _assert_openai_error(self, r_tok_batch, 422, expected_code="context_length_exceeded")
                    self.assertIn("input[1]", err_tok_b["message"])
                    self.assertIn("7,999", err_tok_b["message"])

                    # Homepage HTML, extra="forbid", invalid input objects, and max_labels live enforcement
                    home = await cli.get("/")
                    self.assertEqual(home.status_code, 200)
                    self.assertIn("Amber 150M", home.text)
                    self.assertIn("/v1/classify", home.text)
                    self.assertNotIn("/api/classify", home.text)

                    for bad_body in (
                        {"input": "ok", "labels": ["a", "b"], "unknown": 1},
                        {"input": {"text": "ok", "extra": 1}, "labels": ["a", "b"]},
                        {"input": {"wrong_key": "ok"}, "labels": ["a", "b"]},
                        {"input": "", "labels": ["a", "b"]},
                        {"input": ["ok", "   "], "labels": ["a", "b"]},
                        {"text": "legacy-shape", "labels": ["a", "b"]},
                    ):
                        r_bad = await cli.post("/v1/classify", headers=headers, json=bad_body)
                        _assert_openai_error(self, r_bad, 422, expected_code="validation_error")

                    runtime.apply_patch({"max_labels": 2})
                    too_many = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": "ok", "labels": ["a", "b", "c"]},
                    )
                    _assert_openai_error(self, too_many, 422, expected_code="too_many_labels")

                    # Deeply nested JsonValue (RecursionError / Pydantic recursion guard) -> 422 OpenAI error instead of 500
                    deep_json = ('{"input":{"text":' + ("[" * 1500) + "1" + ("]" * 1500) + '},"labels":["a","b"]}').encode()
                    r_deep = await cli.post(
                        "/v1/classify",
                        headers={**headers, "Content-Type": "application/json"},
                        content=deep_json,
                    )
                    err_deep = _assert_openai_error(self, r_deep, 422)
                    self.assertIn(err_deep["code"], {"validation_error", "nesting_depth_exceeded"})

                    # Body > 256 KiB rejected before auth/JSON parsing (direct & chunked) -> 413 OpenAI error
                    self.assertEqual(MAX_PUBLIC_BODY_BYTES, 256 << 10)
                    oversized = b'{"input":"' + (b"a" * MAX_PUBLIC_BODY_BYTES) + b'","labels":["a","b"]}'
                    r413 = await cli.post(
                        "/v1/classify",
                        headers={"Content-Type": "application/json"},
                        content=oversized,
                    )
                    _assert_openai_error(self, r413, 413, expected_code="payload_too_large")

                    async def chunked_over():
                        yield b'{"input":"'
                        yield b"a" * MAX_PUBLIC_BODY_BYTES
                        yield b'","labels":["a","b"]}'

                    r413_stream = await cli.post(
                        "/v1/classify",
                        headers={"Content-Type": "application/json"},
                        content=chunked_over(),
                    )
                    _assert_openai_error(self, r413_stream, 413, expected_code="payload_too_large")

                    # Worker RuntimeError and item error -> clean 503 OpenAI error
                    r503_item = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": "__ERROR__", "labels": ["a", "b"]},
                    )
                    _assert_openai_error(self, r503_item, 503, expected_code="service_unavailable")

                    def _boom(*_args, **_kwargs):
                        raise RuntimeError("cuda failure")

                    with patch.object(fake_worker, "classify_batch", side_effect=_boom):
                        r503 = await cli.post(
                            "/v1/classify",
                            headers=headers,
                            json={"input": "ok", "labels": ["a", "b"]},
                        )
                        err503 = _assert_openai_error(self, r503, 503, expected_code="service_unavailable")
                        self.assertIn("unavailable", err503["message"].lower())

                    # Backward compatibility with legacy 2-tuple mock workers (usage tokens default to 0)
                    fake_worker.return_tokens = False
                    r_compat = await cli.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": "compat test", "labels": ["a", "b"]},
                    )
                    self.assertEqual(r_compat.status_code, 200)
                    self.assertEqual(r_compat.json()["usage"], {"prompt_tokens": 0, "total_tokens": 0})

                    # Malformed worker payloads (non-mapping probs, NaN, bool score) -> clean 503 OpenAI error
                    for bad_payload in (
                        ("a", None, 10),
                        ("a", {"a": float("nan"), "b": 0.5}, 10),
                        ("a", {"a": float("inf"), "b": 0.5}, 10),
                        ("a", {"a": True, "b": 0.5}, 10),
                    ):
                        with patch.object(fake_worker, "classify_batch", return_value=[("ok", bad_payload)]):
                            r_bad_payload = await cli.post(
                                "/v1/classify",
                                headers=headers,
                                json={"input": "ok", "labels": ["a", "b"]},
                            )
                            _assert_openai_error(self, r_bad_payload, 503, expected_code="service_unavailable")

    async def test_unexpected_500_mid_body_disconnect_and_legacy_classify_fallback(self) -> None:
        runtime = ChoiceRuntime()
        store, public_key = _make_store_with_key()
        headers = {"Authorization": f"Bearer {public_key}"}

        # 1. Unexpected non-RuntimeError exception on /v1/* returns JSON OpenAI 500 envelope
        fake_worker = _FakePublicWorker()
        with patch("app.api.ModelWorker", return_value=fake_worker):
            app = create_app(runtime=runtime, api_key_store=store)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 50015), raise_app_exceptions=False),
                    base_url="http://public.local",
                ) as cli:
                    with patch.object(fake_worker, "classify_batch", side_effect=ZeroDivisionError("unexpected")):
                        r500 = await cli.post(
                            "/v1/classify",
                            headers=headers,
                            json={"input": "ok", "labels": ["a", "b"]},
                        )
                        _assert_openai_error(self, r500, 500, expected_code="internal_error")

                # 2. Mid-body client disconnect in _PublicClassifyLimitMiddleware returns 499 OpenAI error
                sent: list[dict[str, object]] = []
                step = 0

                async def disc_receive():
                    nonlocal step
                    if step == 0:
                        step = 1
                        return {"type": "http.request", "body": b'{"input":', "more_body": True}
                    return {"type": "http.disconnect"}

                async def disc_send(msg):
                    sent.append(msg)

                scope = {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": "POST",
                    "path": "/v1/classify",
                    "raw_path": b"/v1/classify",
                    "query_string": b"",
                    "headers": [
                        (b"authorization", f"Bearer {public_key}".encode("ascii")),
                        (b"content-type", b"application/json"),
                    ],
                    "client": ("127.0.0.1", 51113),
                    "server": ("127.0.0.1", 8000),
                }
                await app(scope, disc_receive, disc_send)
                self.assertEqual(sent[0]["status"], 499)
                disc_body = json.loads(sent[1]["body"].decode("utf-8"))
                self.assertEqual(disc_body["error"]["code"], "request_cancelled")

        # 3. Legacy classify()-only mock worker short-circuits on WorkerInputTooLongError
        called_texts: list[str] = []

        class _LegacyClassifyOnlyWorker:
            idle_timeout = 180.0

            def is_loaded(self) -> bool:
                return True

            def set_idle_timeout(self, _t: float) -> None:
                pass

            def classify(self, text: str, labels, task: str = "decision", prompt: str | None = None):
                called_texts.append(text)
                if "__OVER__" in text:
                    raise WorkerInputTooLongError("exceeds 7,999 tokens")
                names = list(labels)
                return names[0], {n: 1.0 / len(names) for n in names}

            def close(self) -> None:
                pass

        with patch("app.api.ModelWorker", return_value=_LegacyClassifyOnlyWorker()):
            app2 = create_app(runtime=ChoiceRuntime(), api_key_store=store)
            async with app2.router.lifespan_context(app2):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app2, client=("127.0.0.1", 50016)),
                    base_url="http://public.local",
                ) as cli2:
                    r_sc = await cli2.post(
                        "/v1/classify",
                        headers=headers,
                        json={"input": ["__OVER__", "should-not-run"], "labels": ["a", "b"]},
                    )
                    err_sc = _assert_openai_error(self, r_sc, 422, expected_code="context_length_exceeded")
                    self.assertIn("input[0]", err_sc["message"])
                    self.assertEqual(called_texts, ["__OVER__"])

    async def test_middleware_invalid_content_length_returns_openai_400(self) -> None:
        runtime = ChoiceRuntime()
        fake_worker = _FakePublicWorker()
        store, public_key = _make_store_with_key()

        with patch("app.api.ModelWorker", return_value=fake_worker):
            app = create_app(runtime=runtime, api_key_store=store)
            async with app.router.lifespan_context(app):
                sent: list[dict[str, object]] = []

                async def receive():
                    return {"type": "http.request", "body": b"{}", "more_body": False}

                async def send(msg):
                    sent.append(msg)

                scope = {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": "POST",
                    "path": "/v1/classify",
                    "raw_path": b"/v1/classify",
                    "query_string": b"",
                    "headers": [
                        (b"authorization", f"Bearer {public_key}".encode("ascii")),
                        (b"content-length", b"not-a-number"),
                    ],
                    "client": ("127.0.0.1", 51112),
                    "server": ("127.0.0.1", 8000),
                }
                await app(scope, receive, send)
                self.assertEqual(sent[0]["status"], 400)
                payload = json.loads(sent[1]["body"].decode("utf-8"))
                self.assertEqual(payload["error"]["type"], "invalid_request_error")
                self.assertEqual(payload["error"]["code"], "invalid_content_length")

    async def test_lifespan_offloads_start_and_close_without_blocking_event_loop(self) -> None:
        start_entered = threading.Event()
        allow_start_finish = threading.Event()
        close_entered = threading.Event()
        allow_close_finish = threading.Event()

        class _BlockingLifecycleWorker(_FakePublicWorker):
            def start(self) -> None:
                start_entered.set()
                allow_start_finish.wait(timeout=5.0)
                super().start()

            def close(self) -> None:
                close_entered.set()
                allow_close_finish.wait(timeout=5.0)
                super().close()

        runtime = ChoiceRuntime(idle_timeout_seconds=0, worker_idle_timeout=0.0)
        worker = _BlockingLifecycleWorker(idle_timeout=0.0)
        store, _ = _make_store_with_key()

        with patch("app.api.ModelWorker", return_value=worker):
            app = create_app(runtime=runtime, api_key_store=store)
            cm = app.router.lifespan_context(app)

            enter_task = asyncio.create_task(cm.__aenter__())
            try:
                for _ in range(100):
                    if start_entered.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(start_entered.is_set())

                ticks = 0
                for _ in range(5):
                    await asyncio.sleep(0.01)
                    ticks += 1
                self.assertEqual(ticks, 5)
                self.assertFalse(enter_task.done())
            finally:
                allow_start_finish.set()
            await asyncio.wait_for(enter_task, timeout=2.0)
            self.assertTrue(worker.is_loaded())

            exit_task = asyncio.create_task(cm.__aexit__(None, None, None))
            try:
                for _ in range(100):
                    if close_entered.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(close_entered.is_set())

                ticks = 0
                for _ in range(5):
                    await asyncio.sleep(0.01)
                    ticks += 1
                self.assertEqual(ticks, 5)
                self.assertFalse(exit_task.done())
            finally:
                allow_close_finish.set()
            await asyncio.wait_for(exit_task, timeout=2.0)
            self.assertFalse(worker.is_loaded())

    async def test_public_classify_disconnect_and_cancel_propagate_cancel_event(self) -> None:
        entered = threading.Event()
        seen_cancel_events: list[threading.Event] = []
        seen_planes: list[str] = []

        class _CancellableWorker(_FakePublicWorker):
            def classify_batch(
                self,
                items,
                *,
                cancel_event: threading.Event | None = None,
                plane: str = "interactive",
            ):
                assert cancel_event is not None
                seen_cancel_events.append(cancel_event)
                seen_planes.append(plane)
                entered.set()
                if cancel_event.wait(timeout=5.0):
                    raise WorkerCancelledError("cancelled")
                return super().classify_batch(items)

        runtime = ChoiceRuntime()
        worker = _CancellableWorker()
        store, public_key = _make_store_with_key()
        with patch("app.api.ModelWorker", return_value=worker):
            app = create_app(runtime=runtime, api_key_store=store)
            async with app.router.lifespan_context(app):
                body_bytes = json.dumps({"input": ["slow-1", "slow-2"], "labels": ["a", "b"]}).encode("utf-8")
                disconnect_trigger = asyncio.Event()
                step = 0
                sent_messages: list[dict[str, object]] = []

                async def receive():
                    nonlocal step
                    if step == 0:
                        step = 1
                        return {"type": "http.request", "body": body_bytes, "more_body": False}
                    await disconnect_trigger.wait()
                    return {"type": "http.disconnect"}

                async def send(msg):
                    sent_messages.append(msg)

                scope = {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": "POST",
                    "path": "/v1/classify",
                    "raw_path": b"/v1/classify",
                    "query_string": b"",
                    "headers": [
                        (b"authorization", f"Bearer {public_key}".encode("ascii")),
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body_bytes)).encode("ascii")),
                    ],
                    "client": ("127.0.0.1", 51111),
                    "server": ("127.0.0.1", 8000),
                }

                req_task = asyncio.create_task(app(scope, receive, send))
                for _ in range(100):
                    if entered.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(entered.is_set())
                self.assertEqual(seen_planes, ["interactive"])
                self.assertFalse(seen_cancel_events[0].is_set())

                disconnect_trigger.set()
                await asyncio.wait_for(req_task, timeout=2.0)
                self.assertTrue(seen_cancel_events[0].is_set())
                self.assertEqual(sent_messages[0]["status"], 499)
                body = json.loads(sent_messages[1]["body"].decode("utf-8"))
                self.assertEqual(body["error"]["type"], "request_cancelled")
                self.assertEqual(body["error"]["code"], "request_cancelled")


if __name__ == "__main__":
    unittest.main()
