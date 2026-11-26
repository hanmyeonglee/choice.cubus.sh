"""Backward-compatibility and boundary tests for the public Choice API (:8000)."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import unittest
from unittest.mock import patch

import httpx

from app.api import MAX_PUBLIC_BODY_BYTES, create_app
from app.runtime import MODEL_ID, ChoiceRuntime
from app.worker import WorkerCancelledError, WorkerInputTooLongError


PUBLIC_KEY = "choice-public-api-key-abcdef123456"


class _FakePublicWorker:
    def __init__(self, idle_timeout: float = 180.0) -> None:
        self.idle_timeout = float(idle_timeout)
        self.loaded = False
        self.calls: list[dict[str, object]] = []

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
            names = list(labels)
            prob = 1.0 / len(names)
            results.append(("ok", (names[0], {name: prob for name in names})))
        return results

    def classify(self, text: str, labels, task: str = "decision", prompt: str | None = None):
        status, payload = self.classify_batch([(text, labels, task, prompt)])[0]
        if status == "ok":
            return payload
        if status == "input_too_long":
            raise WorkerInputTooLongError(payload)
        raise RuntimeError(payload)

    def close(self) -> None:
        self.loaded = False


class PublicAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_public_classify_healthz_and_limits(self) -> None:
        runtime = ChoiceRuntime()
        fake_worker = _FakePublicWorker()
        headers = {"Authorization": f"Bearer {PUBLIC_KEY}"}

        with patch.dict(os.environ, {"API_KEYS": PUBLIC_KEY}, clear=False):
            with patch("app.api.ModelWorker", return_value=fake_worker):
                app = create_app(runtime=runtime)
                async with app.router.lifespan_context(app):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 50010)),
                        base_url="http://public.local",
                    ) as cli:
                        # /api/healthz
                        hz = await cli.get("/api/healthz")
                        self.assertEqual(hz.status_code, 200)
                        self.assertEqual(hz.json(), {"ok": True, "model_loaded": False, "model": MODEL_ID})

                        # Missing auth -> 401
                        r401 = await cli.post("/api/classify", json={"text": "hello", "labels": ["a", "b"]})
                        self.assertEqual(r401.status_code, 401)

                        # String text + dict labels + prompt + task
                        r_dict = await cli.post(
                            "/api/classify",
                            headers=headers,
                            json={
                                "text": "I was charged twice.",
                                "task": "department",
                                "prompt": "Which team handles this?",
                                "labels": {"billing": "Charges and refunds", "support": None},
                            },
                        )
                        self.assertEqual(r_dict.status_code, 200)
                        self.assertEqual(r_dict.json()["choice"], "billing")
                        self.assertEqual(r_dict.json()["probabilities"], {"billing": 0.5, "support": 0.5})

                        # Healthz reflects model_loaded=True after request
                        hz_loaded = await cli.get("/api/healthz")
                        self.assertEqual(hz_loaded.json(), {"ok": True, "model_loaded": True, "model": MODEL_ID})

                        # Dict and list text inputs are serialized cleanly
                        r_obj = await cli.post(
                            "/api/classify",
                            headers=headers,
                            json={"text": {"cmd": "git status"}, "labels": ["safe", "unsafe"]},
                        )
                        self.assertEqual(r_obj.status_code, 200)
                        self.assertEqual(fake_worker.calls[-1]["text"], '{"cmd":"git status"}')

                        r_arr = await cli.post(
                            "/api/classify",
                            headers=headers,
                            json={"text": ["step1", "step2"], "labels": ["ok", "fail"]},
                        )
                        self.assertEqual(r_arr.status_code, 200)
                        self.assertEqual(fake_worker.calls[-1]["text"], '["step1","step2"]')

                        # Default runtime max_text_chars=8000 allows 8000 and rejects 8001
                        r_8000 = await cli.post(
                            "/api/classify",
                            headers=headers,
                            json={"text": "x" * 8000, "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_8000.status_code, 200)

                        r_8001 = await cli.post(
                            "/api/classify",
                            headers=headers,
                            json={"text": "x" * 8001, "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_8001.status_code, 422)

                        # Live raise to max_text_chars=16000 allows 16000; schema rejects 16001
                        runtime.apply_patch({"max_text_chars": 16000})
                        r_16000 = await cli.post(
                            "/api/classify",
                            headers=headers,
                            json={"text": "x" * 16000, "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_16000.status_code, 200)

                        r_16001 = await cli.post(
                            "/api/classify",
                            headers=headers,
                            json={"text": "x" * 16001, "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_16001.status_code, 422)

                        # Token-budget error from worker -> 422
                        r_tok = await cli.post(
                            "/api/classify",
                            headers=headers,
                            json={"text": "__OVER_TOKENS__", "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_tok.status_code, 422)
                        self.assertIn("7,999", r_tok.json()["detail"])

                        # Homepage HTML & extra="forbid" & max_labels live enforcement
                        home = await cli.get("/")
                        self.assertEqual(home.status_code, 200)
                        self.assertIn("Amber 150M", home.text)

                        extra_field = await cli.post(
                            "/api/classify",
                            headers=headers,
                            json={"text": "ok", "labels": ["a", "b"], "unknown": 1},
                        )
                        self.assertEqual(extra_field.status_code, 422)

                        runtime.apply_patch({"max_labels": 2})
                        too_many = await cli.post(
                            "/api/classify",
                            headers=headers,
                            json={"text": "ok", "labels": ["a", "b", "c"]},
                        )
                        self.assertEqual(too_many.status_code, 422)

                        # Deeply nested JsonValue (RecursionError) -> 422 instead of 500
                        deep_json = ('{"text":' + ("[" * 1500) + "1" + ("]" * 1500) + ',"labels":["a","b"]}').encode()
                        r_deep = await cli.post(
                            "/api/classify",
                            headers={**headers, "Content-Type": "application/json"},
                            content=deep_json,
                        )
                        self.assertEqual(r_deep.status_code, 422)

                        # Body > 128 KiB rejected before auth/JSON parsing (direct & chunked) -> 413
                        oversized = b'{"text":"' + (b"a" * MAX_PUBLIC_BODY_BYTES) + b'","labels":["a","b"]}'
                        r413 = await cli.post(
                            "/api/classify",
                            headers={"Content-Type": "application/json"},
                            content=oversized,
                        )
                        self.assertEqual(r413.status_code, 413)

                        async def chunked_over():
                            yield b'{"text":"'
                            yield b"a" * MAX_PUBLIC_BODY_BYTES
                            yield b'","labels":["a","b"]}'

                        r413_stream = await cli.post(
                            "/api/classify",
                            headers={"Content-Type": "application/json"},
                            content=chunked_over(),
                        )
                        self.assertEqual(r413_stream.status_code, 413)

                        # Worker RuntimeError -> clean JSON 503
                        def _boom(*_args, **_kwargs):
                            raise RuntimeError("cuda failure")

                        with patch.object(fake_worker, "classify", side_effect=_boom):
                            r503 = await cli.post(
                                "/api/classify",
                                headers=headers,
                                json={"text": "ok", "labels": ["a", "b"]},
                            )
                            self.assertEqual(r503.status_code, 503)
                            self.assertIn("unavailable", r503.json()["detail"].lower())

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

        with patch.dict(os.environ, {"API_KEYS": PUBLIC_KEY}, clear=False):
            with patch("app.api.ModelWorker", return_value=worker):
                app = create_app(runtime=runtime)
                cm = app.router.lifespan_context(app)

                enter_task = asyncio.create_task(cm.__aenter__())
                try:
                    # Wait until worker.start() is running in thread pool
                    for _ in range(100):
                        if start_entered.is_set():
                            break
                        await asyncio.sleep(0.01)
                    self.assertTrue(start_entered.is_set())

                    # Event loop MUST remain responsive while worker.start() blocks!
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

                    # Event loop MUST remain responsive while worker.close() blocks!
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
            def classify(
                self,
                text: str,
                labels,
                task: str = "decision",
                prompt: str | None = None,
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
                return super().classify(text, labels, task=task, prompt=prompt)

        runtime = ChoiceRuntime()
        worker = _CancellableWorker()
        with patch.dict(os.environ, {"API_KEYS": PUBLIC_KEY}, clear=False):
            with patch("app.api.ModelWorker", return_value=worker):
                app = create_app(runtime=runtime)
                async with app.router.lifespan_context(app):
                    # Simulate raw ASGI client sending body then http.disconnect while worker is blocked
                    body_bytes = json.dumps({"text": "slow", "labels": ["a", "b"]}).encode("utf-8")
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
                        "path": "/api/classify",
                        "raw_path": b"/api/classify",
                        "query_string": b"",
                        "headers": [
                            (b"authorization", f"Bearer {PUBLIC_KEY}".encode("ascii")),
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

                    # Trigger HTTP client disconnect -> cancel_event must be set promptly
                    disconnect_trigger.set()
                    await asyncio.wait_for(req_task, timeout=2.0)
                    self.assertTrue(seen_cancel_events[0].is_set())
                    self.assertEqual(sent_messages[0]["status"], 499)


if __name__ == "__main__":
    unittest.main()
