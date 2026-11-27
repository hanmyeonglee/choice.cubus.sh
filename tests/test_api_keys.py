"""Comprehensive tests for db-broker bootstrap, PostgreSQL API key store, caching, usage batching, and :9090 api_keys endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import email.message
import hashlib
import http.client
import io
import json
import logging
import os
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
from uuid import UUID, uuid4

import httpx
import psycopg

from app.admin import MAX_ADMIN_BODY_BYTES, create_admin_app, load_admin_config
from app.api import create_app
from app.api_keys import (
    APIKeyRecord,
    APIKeyStore,
    APIKeyStoreError,
    InMemoryAPIKeyBackend,
    hash_api_key,
    is_valid_key_format,
)
from app.db import (
    CREATE_API_KEYS_TABLE_SQL,
    DatabaseConfigError,
    PostgresAPIKeyBackend,
    resolve_database_config,
)
from app.runtime import ChoiceRuntime


VALID_ADMIN_TOKEN = "choice-admin-secret-token-123456"


class _FakeWorker:
    def __init__(self, idle_timeout: float = 180.0) -> None:
        self.idle_timeout = float(idle_timeout)
        self.loaded = False

    def is_loaded(self) -> bool:
        return self.loaded

    def start(self) -> None:
        self.loaded = True

    def set_idle_timeout(self, idle_timeout: float) -> None:
        self.idle_timeout = float(idle_timeout)

    def classify_batch(self, items):
        self.loaded = True
        results = []
        for _text, labels, _task, _prompt in items:
            names = list(labels)
            prob = 1.0 / len(names)
            results.append(("ok", (names[0], {name: prob for name in names}, 12)))
        return results

    def classify(self, text: str, labels, task: str = "decision", prompt: str | None = None):
        self.loaded = True
        names = list(labels)
        prob = 1.0 / len(names)
        return names[0], {name: prob for name in names}

    def close(self) -> None:
        self.loaded = False


class _FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._buf = io.BytesIO(body)

    def read(self, amt: int = -1) -> bytes:
        return self._buf.read(amt)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _http_error(code: int, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    hdrs = email.message.Message()
    for k, v in (headers or {}).items():
        hdrs[k] = v
    return urllib.error.HTTPError(
        url="http://db-broker:8080/v1/database",
        code=code,
        msg=f"HTTP {code}",
        hdrs=hdrs,
        fp=io.BytesIO(b'{"error":"boom"}'),
    )


class DatabaseConfigAndPostgresBackendTests(unittest.TestCase):
    def test_resolve_database_config_precedence_and_validation(self) -> None:
        # Neither env set -> fail fast
        with self.assertRaises(DatabaseConfigError):
            resolve_database_config({})

        # Invalid DATABASE_URL scheme -> fail without echoing secret
        secret_bad = "mysql://user:super-secret-pass@db:3306/choice"
        with self.assertRaises(DatabaseConfigError) as cm:
            resolve_database_config({"DATABASE_URL": secret_bad})
        self.assertNotIn("super-secret-pass", str(cm.exception))

        # Standalone DATABASE_URL works when DB_BROKER_URL is unset
        cfg = resolve_database_config({"DATABASE_URL": "postgresql://choice:dev@postgres:5432/choice?sslmode=disable"})
        self.assertEqual(cfg, {"url": "postgresql://choice:dev@postgres:5432/choice?sslmode=disable"})

        # Reject non-http(s) DB_BROKER_URL or embedded credentials
        for bad_broker in (
            "file:///etc/passwd",
            "ftp://db-broker:8080/v1/database",
            "http://user:pass@db-broker:8080/v1/database",
            "http://db-broker:8080/v1/database#frag",
        ):
            with self.subTest(bad_broker=bad_broker):
                with self.assertRaises(DatabaseConfigError):
                    resolve_database_config({"DB_BROKER_URL": bad_broker})

    def test_resolve_database_config_retries_on_503_and_network_error_and_never_logs_secrets(self) -> None:
        secret_pw = "super-secret-db-broker-password-987654321"
        secret_url = f"postgresql://svc_choice:{secret_pw}@postgres:5432/svc_choice?sslmode=disable"
        payload = json.dumps(
            {
                "database": "svc_choice",
                "username": "svc_choice",
                "password": secret_pw,
                "host": "postgres",
                "port": 5432,
                "url": secret_url,
            }
        ).encode("utf-8")

        attempts: list[str] = []
        sleeps: list[float] = []

        class _Opener:
            def open(self, req, timeout: float = 10.0):
                attempts.append(f"{req.get_method()} {req.full_url}")
                self_idx = len(attempts)
                if self_idx == 1:
                    raise _http_error(503, {"Retry-After": "1"})
                if self_idx == 2:
                    raise urllib.error.URLError("connection refused")
                if self_idx == 3:
                    raise _http_error(503, {"Retry-After": "999"})  # capped to MAX_RETRY_DELAY_SECONDS
                return _FakeHTTPResponse(payload)

        with self.assertLogs("app.db", level=logging.INFO) as log_cm:
            cfg = resolve_database_config(
                {
                    "DB_BROKER_URL": "http://db-broker:8080/v1/database",
                    "DATABASE_URL": "postgresql://ignored@localhost/ignored",
                },
                sleep_fn=sleeps.append,
                jitter_fn=lambda _a, _b: 0.1,
                opener=_Opener(),
            )

        self.assertEqual(cfg["url"], secret_url)
        self.assertEqual(len(attempts), 4)
        self.assertEqual(sleeps, [1.1, 1.1, 10.1])
        combined_logs = "\n".join(log_cm.output)
        self.assertIn("svc_choice", combined_logs)
        self.assertNotIn(secret_pw, combined_logs)
        self.assertNotIn(secret_url, combined_logs)

    def test_resolve_database_config_fails_fast_on_400_403_409_and_redirects(self) -> None:
        for code in (302, 400, 403, 409):
            with self.subTest(code=code):
                calls = 0
                sleeps: list[float] = []

                class _FailOpener:
                    def open(self, _req, timeout: float = 10.0):
                        nonlocal calls
                        calls += 1
                        raise _http_error(code)

                with self.assertRaises(DatabaseConfigError) as cm:
                    resolve_database_config(
                        {"DB_BROKER_URL": "http://db-broker:8080/v1/database"},
                        sleep_fn=sleeps.append,
                        opener=_FailOpener(),
                    )
                self.assertEqual(calls, 1)
                self.assertEqual(sleeps, [])
                self.assertIn(str(code), str(cm.exception))

    def test_postgres_backend_queries_and_sanitized_errors(self) -> None:
        executed: list[tuple[str, object]] = []
        now = datetime(2026, 11, 26, 12, 0, 0, tzinfo=timezone.utc)
        key_id = uuid4()
        key_hash = hash_api_key("chk_" + "a" * 43)

        class _FakeCursor:
            def __init__(self) -> None:
                self._next_one: object = None
                self._next_all: list[object] = []

            def execute(self, sql: str, params: object = None) -> None:
                executed.append((sql.strip(), params))
                if "WHERE key_hash = %s" in sql:
                    self._next_one = (key_id, key_hash, "active", None)
                elif "FROM api_keys" in sql and "ORDER BY" in sql:
                    self._next_all = [
                        (key_id, "svc", "ops", "chk_aaaaaaaa", "active", None, None, 3, "admin", now, None)
                    ]
                elif "INSERT INTO api_keys" in sql:
                    self._next_one = (
                        key_id, "svc", "ops", "chk_aaaaaaaa", "active", None, None, 0, "admin", now, None
                    )
                elif "SET status = 'revoked'" in sql:
                    self._next_one = (
                        key_id, "svc", "ops", "chk_aaaaaaaa", "revoked", None, None, 3, "admin", now, now, key_hash
                    )
                elif "DELETE FROM api_keys" in sql:
                    self._next_one = (key_hash,)

            def executemany(self, sql: str, params_seq: object) -> None:
                executed.append((sql.strip(), list(params_seq)))  # type: ignore[arg-type]

            def fetchone(self):
                return self._next_one

            def fetchall(self):
                return self._next_all

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        class _FakeTx:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        class _FakeConn:
            def cursor(self):
                return _FakeCursor()

            def transaction(self):
                return _FakeTx()

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        class _FakePool:
            def __init__(self) -> None:
                self.waited = False
                self.closed = False
                self.raise_on_conn = False

            def wait(self, timeout: float = 15.0) -> None:
                self.waited = True

            def connection(self):
                if self.raise_on_conn:
                    raise psycopg.OperationalError("connection failed for password=secret-leak")
                return _FakeConn()

            def close(self) -> None:
                self.closed = True

        pool = _FakePool()
        backend = PostgresAPIKeyBackend(pool=pool)
        backend.initialize()
        self.assertTrue(pool.waited)
        self.assertTrue(any(CREATE_API_KEYS_TABLE_SQL.strip() in sql for sql, _ in executed))

        auth = backend.lookup_by_hash(key_hash)
        self.assertIsNotNone(auth)
        assert auth is not None
        self.assertEqual(auth.id, key_id)

        listed = backend.list_keys()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].request_count, 3)

        rec = APIKeyRecord(
            id=key_id,
            name="svc",
            owner="ops",
            key_prefix="chk_aaaaaaaa",
            status="active",
            expires_at=None,
            last_used_at=None,
            request_count=0,
            created_by="admin",
            created_at=now,
            revoked_at=None,
        )
        created = backend.create_key(rec, key_hash)
        self.assertEqual(created.id, key_id)

        revoked = backend.revoke_key(key_id, now)
        self.assertIsNotNone(revoked)
        assert revoked is not None
        self.assertEqual(revoked[0].status, "revoked")
        self.assertEqual(revoked[1], key_hash)

        deleted_hash = backend.delete_key(key_id)
        self.assertEqual(deleted_hash, key_hash)

        backend.flush_usage_batch([(key_id, 5, now)])
        self.assertTrue(any("request_count = request_count + %s" in sql for sql, _ in executed))

        # OperationalError converted to sanitized APIKeyStoreError (never leaks password)
        pool.raise_on_conn = True
        with self.assertRaises(APIKeyStoreError) as err_cm:
            backend.lookup_by_hash(key_hash)
        self.assertNotIn("secret-leak", str(err_cm.exception))

        backend.close()
        self.assertTrue(pool.closed)

    def test_resolve_database_config_retries_http_exception_and_suppresses_raw_body_cause(self) -> None:
        secret_url = "postgresql://svc_choice:pw123@postgres:5432/svc_choice?sslmode=disable"
        attempts = 0
        sleeps: list[float] = []

        class _FlakyHTTPOpener:
            def open(self, _req, timeout: float = 10.0):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise http.client.IncompleteRead(b'{"url":')
                if attempts == 2:
                    raise http.client.BadStatusLine("EOF")
                return _FakeHTTPResponse(json.dumps({"url": secret_url}).encode("utf-8"))

        cfg = resolve_database_config(
            {"DB_BROKER_URL": "http://db-broker:8080/v1/database"},
            sleep_fn=sleeps.append,
            jitter_fn=lambda _a, _b: 0.0,
            opener=_FlakyHTTPOpener(),
        )
        self.assertEqual(cfg["url"], secret_url)
        self.assertEqual(attempts, 3)
        self.assertEqual(len(sleeps), 2)

        # Malformed UTF-8 payload must not attach raw bytes to __cause__
        class _BadBytesOpener:
            def open(self, _req, timeout: float = 10.0):
                return _FakeHTTPResponse(b"\xffsecret-raw-password")

        with self.assertRaises(DatabaseConfigError) as err_cm:
            resolve_database_config(
                {"DB_BROKER_URL": "http://db-broker:8080/v1/database"},
                opener=_BadBytesOpener(),
            )
        self.assertIsNone(err_cm.exception.__cause__)
        self.assertNotIn("secret-raw-password", str(err_cm.exception))

    def test_postgres_backend_configures_pool_timeouts_keepalives_and_health_check(self) -> None:
        from psycopg_pool import ConnectionPool

        captured: dict[str, object] = {}

        def _fake_pool_ctor(**kwargs):
            captured.update(kwargs)
            return object()

        with patch("app.db.ConnectionPool", side_effect=_fake_pool_ctor) as mock_pool_cls:
            mock_pool_cls.check_connection = ConnectionPool.check_connection
            PostgresAPIKeyBackend("postgresql://svc_choice:pw@postgres:5432/svc_choice?sslmode=disable")

        self.assertEqual(captured["min_size"], 1)
        self.assertEqual(captured["max_size"], 8)
        self.assertIs(captured["check"], ConnectionPool.check_connection)
        conn_kwargs = captured["kwargs"]
        assert isinstance(conn_kwargs, dict)
        self.assertEqual(conn_kwargs["connect_timeout"], 5)
        self.assertEqual(conn_kwargs["options"], "-c statement_timeout=5000")
        self.assertEqual(conn_kwargs["keepalives"], 1)
        self.assertEqual(conn_kwargs["keepalives_idle"], 30)
        self.assertEqual(conn_kwargs["keepalives_interval"], 10)
        self.assertEqual(conn_kwargs["keepalives_count"], 3)


class APIKeyStoreBehaviorTests(unittest.TestCase):
    def test_key_format_hashing_cache_hit_ttl_lru_and_no_negative_cache(self) -> None:
        clock_utc = [datetime(2026, 11, 26, 12, 0, 0, tzinfo=timezone.utc)]
        clock_mono = [100.0]

        backend = InMemoryAPIKeyBackend()
        store = APIKeyStore(
            backend,
            cache_ttl_seconds=15.0,
            cache_max_size=2,
            now_fn=lambda: clock_utc[0],
            monotonic_fn=lambda: clock_mono[0],
        )

        key1, rec1 = store.create_key(name="k1", owner="alice", expires_in_days=30, created_by="root")
        self.assertTrue(key1.startswith("chk_"))
        self.assertGreaterEqual(len(key1), 40)
        self.assertTrue(is_valid_key_format(key1))
        self.assertEqual(rec1["key_prefix"], key1[:12])
        self.assertNotIn("key_hash", rec1)
        self.assertNotIn("api_key", rec1)

        # Plaintext is never stored in backend; only SHA-256 hex digest is stored
        expected_hash = hashlib.sha256(key1.encode("utf-8")).hexdigest()
        self.assertIn(expected_hash, backend._hash_to_id)
        self.assertNotIn(key1, backend._hash_to_id)

        # Malformed keys are rejected before hitting backend
        self.assertIsNone(store.verify_key("not-a-chk-key"))
        self.assertIsNone(store.verify_key("chk_short"))
        self.assertIsNone(store.verify_key("chk_" + "x" * 200))
        self.assertEqual(backend.lookup_calls, 0)

        # Unknown valid-format keys are NEVER negative-cached (prevents OOM)
        for idx in range(50):
            self.assertIsNone(store.verify_key(f"chk_{idx:043d}"))
        self.assertEqual(len(store._cache), 0)
        self.assertEqual(backend.lookup_calls, 50)

        # First valid verify populates cache; next 10 hits do zero DB lookups
        self.assertEqual(str(store.verify_key(key1)), rec1["id"])
        self.assertEqual(backend.lookup_calls, 51)
        for _ in range(10):
            self.assertEqual(str(store.verify_key(key1)), rec1["id"])
        self.assertEqual(backend.lookup_calls, 51)

        # After 15s TTL elapses, next verify queries DB again
        clock_mono[0] += 15.1
        self.assertEqual(str(store.verify_key(key1)), rec1["id"])
        self.assertEqual(backend.lookup_calls, 52)

        # Bounded LRU eviction (max_size=2)
        key2, _ = store.create_key(name="k2")
        key3, _ = store.create_key(name="k3")
        self.assertIsNotNone(store.verify_key(key2))
        self.assertIsNotNone(store.verify_key(key3))
        self.assertEqual(len(store._cache), 2)
        self.assertNotIn(expected_hash, store._cache)

        # Advancing UTC clock past 30-day expiry rejects key immediately and marks effective status "expired"
        store.verify_key(key1)  # re-warm cache
        clock_utc[0] += timedelta(days=30, seconds=1)
        self.assertIsNone(store.verify_key(key1))
        listed = {item["id"]: item for item in store.list_keys()}
        self.assertEqual(listed[rec1["id"]]["status"], "expired")

    def test_immediate_invalidation_and_concurrent_revoke_race_fence(self) -> None:
        lookup_entered = threading.Event()
        release_lookup = threading.Event()

        class _RacingBackend(InMemoryAPIKeyBackend):
            def __init__(self) -> None:
                super().__init__()
                self.block_next_lookup = False

            def lookup_by_hash(self, key_hash: str):
                result = super().lookup_by_hash(key_hash)
                if self.block_next_lookup:
                    self.block_next_lookup = False
                    lookup_entered.set()
                    release_lookup.wait(timeout=5.0)
                return result

        backend = _RacingBackend()
        store = APIKeyStore(backend, cache_ttl_seconds=15.0)
        raw_key, rec = store.create_key(name="race-key")
        key_id = UUID(rec["id"])

        # Start an auth lookup that reads active state, then pauses before returning
        backend.block_next_lookup = True
        auth_results: list[UUID | None] = []

        def _auth_worker() -> None:
            auth_results.append(store.verify_key(raw_key))

        t = threading.Thread(target=_auth_worker)
        t.start()
        self.assertTrue(lookup_entered.wait(timeout=2.0))

        # Admin revokes the key while the DB read is in flight
        revoked = store.revoke_key(key_id)
        self.assertIsNotNone(revoked)
        assert revoked is not None
        self.assertEqual(revoked["status"], "revoked")

        # Release the in-flight auth read: generation fence forces re-check -> rejected & NOT cached!
        release_lookup.set()
        t.join(timeout=2.0)
        self.assertEqual(auth_results, [None])
        self.assertEqual(len(store._cache), 0)
        self.assertIsNone(store.verify_key(raw_key))

    def test_batched_usage_flush_retry_on_failure_and_shutdown_flush(self) -> None:
        clock_utc = [datetime(2026, 11, 26, 12, 0, 0, tzinfo=timezone.utc)]

        class _FlakyFlushBackend(InMemoryAPIKeyBackend):
            def __init__(self) -> None:
                super().__init__()
                self.fail_next_flush = False

            def flush_usage_batch(self, updates):
                if self.fail_next_flush:
                    self.fail_next_flush = False
                    raise APIKeyStoreError("transient db blip")
                return super().flush_usage_batch(updates)

        backend = _FlakyFlushBackend()
        store = APIKeyStore(
            backend,
            flush_interval_seconds=0.05,
            now_fn=lambda: clock_utc[0],
        )
        raw_key, rec = store.create_key(name="usage-key")

        # 3 verifications accumulate in memory without hitting flush_usage_batch
        for _ in range(3):
            self.assertIsNotNone(store.verify_key(raw_key))
        self.assertEqual(backend.flush_calls, 0)

        # list_keys() immediately reflects live pending usage
        listed = store.list_keys()
        self.assertEqual(listed[0]["request_count"], 3)
        self.assertEqual(listed[0]["last_used_at"], "2026-11-26T12:00:00Z")

        # Failed flush re-queues pending usage so counts are never lost
        backend.fail_next_flush = True
        self.assertEqual(store.flush_usage(), 0)
        self.assertEqual(store.list_keys()[0]["request_count"], 3)

        # Next flush persists cleanly
        self.assertEqual(store.flush_usage(), 1)
        self.assertEqual(backend.flush_calls, 1)
        self.assertEqual(backend.list_keys()[0].request_count, 3)

        # Background flusher + graceful close() flushes final increments
        clock_utc[0] += timedelta(seconds=10)
        store.verify_key(raw_key)
        store.start_background_flush()
        time.sleep(0.12)
        self.assertEqual(backend.list_keys()[0].request_count, 4)

        clock_utc[0] += timedelta(seconds=10)
        store.verify_key(raw_key)
        store.close()
        self.assertEqual(backend.list_keys()[0].request_count, 5)
        self.assertTrue(backend.closed)

    def test_inflight_flush_prevents_transient_undercount_and_avoids_duplicate_shutdown_flush(self) -> None:
        flush_entered = threading.Event()
        release_flush = threading.Event()
        clock_utc = [datetime(2026, 11, 26, 12, 0, 0, tzinfo=timezone.utc)]

        class _SlowFlushBackend(InMemoryAPIKeyBackend):
            def __init__(self) -> None:
                super().__init__()
                self.block_next_flush = False
                self.always_fail_flush = False
                self.flush_attempts = 0

            def flush_usage_batch(self, updates):
                self.flush_attempts += 1
                if self.always_fail_flush:
                    raise APIKeyStoreError("db down")
                if self.block_next_flush:
                    self.block_next_flush = False
                    flush_entered.set()
                    release_flush.wait(timeout=5.0)
                return super().flush_usage_batch(updates)

        backend = _SlowFlushBackend()
        store = APIKeyStore(backend, now_fn=lambda: clock_utc[0])
        raw_key, _rec = store.create_key(name="inflight-key")
        self.assertIsNotNone(store.verify_key(raw_key))
        self.assertIsNotNone(store.verify_key(raw_key))

        backend.block_next_flush = True
        flusher = threading.Thread(target=store.flush_usage)
        flusher.start()
        self.assertTrue(flush_entered.wait(timeout=2.0))

        # While flush is in-flight (pre-commit), list_keys() must still see request_count=2!
        mid_list = store.list_keys()
        self.assertEqual(mid_list[0]["request_count"], 2)
        self.assertEqual(mid_list[0]["last_used_at"], "2026-11-26T12:00:00Z")

        release_flush.set()
        flusher.join(timeout=2.0)
        self.assertEqual(store.list_keys()[0]["request_count"], 2)

        # Shutdown flush followed immediately by close() must not double-attempt a failing DB flush
        store.verify_key(raw_key)
        backend.always_fail_flush = True
        attempts_before = backend.flush_attempts
        store.stop_background_flush(flush=True)
        store.close()
        self.assertEqual(backend.flush_attempts, attempts_before + 1)


class PublicAndAdminAPIKeyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_end_to_end_lifecycle_headers_env_removal_and_immediate_revocation(self) -> None:
        clock_utc = [datetime(2026, 11, 26, 12, 0, 0, tzinfo=timezone.utc)]
        backend = InMemoryAPIKeyBackend()
        store = APIKeyStore(backend, cache_ttl_seconds=15.0, now_fn=lambda: clock_utc[0])
        runtime = ChoiceRuntime()
        fake_worker = _FakeWorker()
        admin_cfg = load_admin_config({"ADMIN_API_TOKEN": VALID_ADMIN_TOKEN})

        # Even if legacy API_KEYS is present in environment, it MUST NOT authenticate!
        legacy_env_key = "chk_" + "z" * 43
        with patch.dict(os.environ, {"API_KEYS": legacy_env_key, "API_KEY": legacy_env_key}, clear=False):
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
                        adm_headers = {"Authorization": f"Bearer {VALID_ADMIN_TOKEN}"}

                        # 1. Legacy .env API_KEYS must be rejected with 401
                        r_legacy_bearer = await pub_cli.post(
                            "/v1/classify",
                            headers={"Authorization": f"Bearer {legacy_env_key}"},
                            json={"input": "hello", "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_legacy_bearer.status_code, 401)
                        r_legacy_xapi = await pub_cli.post(
                            "/v1/classify",
                            headers={"X-API-Key": legacy_env_key},
                            json={"input": "hello", "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_legacy_xapi.status_code, 401)
                        r_legacy_models = await pub_cli.get(
                            "/v1/models",
                            headers={"Authorization": f"Bearer {legacy_env_key}"},
                        )
                        self.assertEqual(r_legacy_models.status_code, 401)

                        # 2. Healthz stays unauthenticated
                        hz = await pub_cli.get("/healthz")
                        self.assertEqual(hz.status_code, 200)

                        # 3. Create key via :9090 POST /admin/v1/api-keys
                        create_resp = await adm_cli.post(
                            "/admin/v1/api-keys",
                            headers=adm_headers,
                            json={
                                "name": "Prod Inference Bot",
                                "owner": "ml-team",
                                "expires_in_days": 30,
                                "created_by": "admin@cubus.sh",
                            },
                        )
                        self.assertEqual(create_resp.status_code, 201)
                        self.assertEqual(create_resp.headers.get("cache-control"), "no-store")
                        self.assertEqual(create_resp.headers.get("x-content-type-options"), "nosniff")
                        created_body = create_resp.json()
                        plaintext = created_body["api_key"]
                        record = created_body["record"]
                        self.assertTrue(plaintext.startswith("chk_"))
                        self.assertEqual(record["key_prefix"], plaintext[:12])
                        self.assertEqual(record["name"], "Prod Inference Bot")
                        self.assertEqual(record["owner"], "ml-team")
                        self.assertEqual(record["created_by"], "admin@cubus.sh")
                        self.assertEqual(record["status"], "active")
                        self.assertEqual(record["request_count"], 0)
                        self.assertIsNotNone(record["expires_at"])
                        self.assertIsNone(record["last_used_at"])
                        self.assertIsNone(record["revoked_at"])
                        self.assertNotIn("key_hash", record)

                        # 4. Authenticate using Authorization: Bearer AND X-API-Key
                        r_bearer = await pub_cli.post(
                            "/v1/classify",
                            headers={"Authorization": f"Bearer {plaintext}"},
                            json={"input": "hello", "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_bearer.status_code, 200)

                        r_xapi = await pub_cli.post(
                            "/v1/classify",
                            headers={"X-API-Key": plaintext},
                            json={"input": "hello", "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_xapi.status_code, 200)

                        r_models_ok = await pub_cli.get(
                            "/v1/models",
                            headers={"Authorization": f"Bearer {plaintext}"},
                        )
                        self.assertEqual(r_models_ok.status_code, 200)

                        # Conflicting Bearer + X-API-Key or non-Bearer Authorization -> 401
                        r_conflict = await pub_cli.post(
                            "/v1/classify",
                            headers={
                                "Authorization": f"Bearer {plaintext}",
                                "X-API-Key": "chk_" + "0" * 43,
                            },
                            json={"input": "hello", "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_conflict.status_code, 401)

                        r_basic = await pub_cli.post(
                            "/v1/classify",
                            headers={"Authorization": "Basic abc", "X-API-Key": plaintext},
                            json={"input": "hello", "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_basic.status_code, 401)

                        # 5. List keys shows count=3, never leaks plaintext or key_hash
                        list_resp = await adm_cli.get("/admin/v1/api-keys", headers=adm_headers)
                        self.assertEqual(list_resp.status_code, 200)
                        serialized_list = list_resp.text
                        self.assertNotIn(plaintext, serialized_list)
                        self.assertNotIn(hash_api_key(plaintext), serialized_list)
                        items = list_resp.json()["api_keys"]
                        self.assertEqual(len(items), 1)
                        self.assertEqual(items[0]["request_count"], 3)
                        self.assertEqual(items[0]["last_used_at"], "2026-11-26T12:00:00Z")

                        # 6. Revoke key -> IMMEDIATE 401 on cached key!
                        key_id = record["id"]
                        rev_resp = await adm_cli.post(f"/admin/v1/api-keys/{key_id}/revoke", headers=adm_headers)
                        self.assertEqual(rev_resp.status_code, 200)
                        rev_body = rev_resp.json()
                        self.assertEqual(rev_body["status"], "revoked")
                        self.assertIsNotNone(rev_body["revoked_at"])
                        self.assertNotIn("key_hash", rev_body)

                        r_after_revoke = await pub_cli.post(
                            "/v1/classify",
                            headers={"X-API-Key": plaintext},
                            json={"input": "hello", "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_after_revoke.status_code, 401)

                        r_models_after_revoke = await pub_cli.get(
                            "/v1/models",
                            headers={"Authorization": f"Bearer {plaintext}"},
                        )
                        self.assertEqual(r_models_after_revoke.status_code, 401)

                        # 7. Create a second key (never expires), warm cache, then DELETE -> IMMEDIATE 401!
                        create2 = await adm_cli.post(
                            "/admin/v1/api-keys",
                            headers=adm_headers,
                            json={"name": "Disposable Key", "expires_in_days": None},
                        )
                        self.assertEqual(create2.status_code, 201)
                        key2 = create2.json()["api_key"]
                        key2_id = create2.json()["record"]["id"]
                        self.assertIsNone(create2.json()["record"]["expires_at"])

                        r_k2_ok = await pub_cli.post(
                            "/v1/classify",
                            headers={"X-API-Key": key2},
                            json={"input": "hello", "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_k2_ok.status_code, 200)

                        del_resp = await adm_cli.delete(f"/admin/v1/api-keys/{key2_id}", headers=adm_headers)
                        self.assertEqual(del_resp.status_code, 200)
                        self.assertEqual(del_resp.json(), {"deleted": True, "id": key2_id})

                        r_k2_after_del = await pub_cli.post(
                            "/v1/classify",
                            headers={"X-API-Key": key2},
                            json={"input": "hello", "labels": ["a", "b"]},
                        )
                        self.assertEqual(r_k2_after_del.status_code, 401)

                        # 404 on unknown UUIDs, 400 on malformed UUIDs
                        missing_id = str(uuid4())
                        self.assertEqual(
                            (await adm_cli.post(f"/admin/v1/api-keys/{missing_id}/revoke", headers=adm_headers)).status_code,
                            404,
                        )
                        self.assertEqual(
                            (await adm_cli.delete(f"/admin/v1/api-keys/{missing_id}", headers=adm_headers)).status_code,
                            404,
                        )
                        self.assertEqual(
                            (await adm_cli.post("/admin/v1/api-keys/not-a-uuid/revoke", headers=adm_headers)).status_code,
                            400,
                        )
                        self.assertEqual(
                            (await adm_cli.delete("/admin/v1/api-keys/not-a-uuid", headers=adm_headers)).status_code,
                            400,
                        )

                # Lifespan exit must have flushed pending usage counts into backend!
                stored_records = {str(r.id): r for r in backend.list_keys()}
                self.assertEqual(stored_records[key_id].request_count, 3)

    async def test_admin_api_keys_strict_validation_cidr_bearer_and_db_unavailable(self) -> None:
        backend = InMemoryAPIKeyBackend()
        store = APIKeyStore(backend)
        rt = ChoiceRuntime()
        cfg = load_admin_config({"ADMIN_API_TOKEN": VALID_ADMIN_TOKEN, "ADMIN_ALLOW_CIDR": "172.31.0.2/32"})
        app = create_admin_app(rt, cfg, api_key_store=store)

        # Outside CIDR -> 403 even with valid token & spoofed XFF
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("172.31.0.99", 50000)),
            base_url="http://admin.local",
        ) as bad_peer_cli:
            for method, path in (
                ("GET", "/admin/v1/api-keys"),
                ("POST", "/admin/v1/api-keys"),
                ("POST", f"/admin/v1/api-keys/{uuid4()}/revoke"),
                ("DELETE", f"/admin/v1/api-keys/{uuid4()}"),
            ):
                r = await bad_peer_cli.request(
                    method,
                    path,
                    headers={"Authorization": f"Bearer {VALID_ADMIN_TOKEN}", "X-Forwarded-For": "172.31.0.2"},
                    json={"name": "x"} if method == "POST" and path.endswith("/api-keys") else None,
                )
                self.assertEqual(r.status_code, 403)

        # Allowed CIDR without Bearer -> 401
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("172.31.0.2", 50000)),
            base_url="http://admin.local",
        ) as cli:
            self.assertEqual((await cli.get("/admin/v1/api-keys")).status_code, 401)

            headers = {"Authorization": f"Bearer {VALID_ADMIN_TOKEN}", "Content-Type": "application/json"}
            bad_payloads = [
                "{}",
                '{"name": ""}',
                '{"name": "   "}',
                json.dumps({"name": "a" * 121}),
                json.dumps({"name": "bad\nname"}),
                json.dumps({"name": "bad\u0085nel"}),
                json.dumps({"name": "bad\u009bcsi"}),
                json.dumps({"name": "ok", "owner": "bad\u009bowner"}),
                json.dumps({"name": "ok", "created_by": "bad\u0085user"}),
                json.dumps({"name": "ok", "owner": 123}),
                json.dumps({"name": "ok", "owner": None}),
                json.dumps({"name": "ok", "owner": "o" * 121}),
                json.dumps({"name": "ok", "created_by": "c" * 121}),
                json.dumps({"name": "ok", "created_by": False}),
                json.dumps({"name": "ok", "expires_in_days": 0}),
                json.dumps({"name": "ok", "expires_in_days": -30}),
                json.dumps({"name": "ok", "expires_in_days": 7}),
                json.dumps({"name": "ok", "expires_in_days": 180}),
                json.dumps({"name": "ok", "expires_in_days": 400}),
                json.dumps({"name": "ok", "expires_in_days": True}),
                json.dumps({"name": "ok", "expires_in_days": 30.0}),
                json.dumps({"name": "ok", "expires_in_days": "30"}),
                json.dumps({"name": "ok", "extra_field": 1}),
                '{"name": "a", "name": "b"}',
                "[]",
                "{bad",
            ]
            for raw in bad_payloads:
                with self.subTest(raw=raw[:80]):
                    r = await cli.post("/admin/v1/api-keys", headers=headers, content=raw.encode("utf-8"))
                    self.assertEqual(r.status_code, 400)
                    self.assertIn("error", r.json())

            # Oversized body > 16 KiB -> 413
            oversized = b'{"name":"' + (b"a" * MAX_ADMIN_BODY_BYTES) + b'"}'
            r413 = await cli.post("/admin/v1/api-keys", headers=headers, content=oversized)
            self.assertEqual(r413.status_code, 413)

            # Valid expires_in_days options: 30, 90, 365
            for days in (30, 90, 365):
                r_ok = await cli.post(
                    "/admin/v1/api-keys",
                    headers=headers,
                    json={"name": f"key-{days}", "expires_in_days": days},
                )
                self.assertEqual(r_ok.status_code, 201)

        # Store unavailable (None or raising APIKeyStoreError) -> 503
        app_no_store = create_admin_app(rt, cfg, api_key_store=None)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_no_store, client=("172.31.0.2", 50000)),
            base_url="http://admin.local",
        ) as cli_none:
            headers = {"Authorization": f"Bearer {VALID_ADMIN_TOKEN}"}
            self.assertEqual((await cli_none.get("/admin/v1/api-keys", headers=headers)).status_code, 503)
            self.assertEqual(
                (await cli_none.post("/admin/v1/api-keys", headers=headers, json={"name": "k"})).status_code,
                503,
            )

        # Public API returns 503 when DB lookup raises APIKeyStoreError on cache miss
        class _BrokenBackend(InMemoryAPIKeyBackend):
            def lookup_by_hash(self, key_hash: str):
                raise APIKeyStoreError("db down")

        broken_store = APIKeyStore(_BrokenBackend())
        with patch("app.api.ModelWorker", return_value=_FakeWorker()):
            pub_app = create_app(runtime=ChoiceRuntime(), api_key_store=broken_store)
            async with pub_app.router.lifespan_context(pub_app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=pub_app, client=("127.0.0.1", 50001)),
                    base_url="http://public.local",
                ) as pub_cli:
                    r503 = await pub_cli.post(
                        "/v1/classify",
                        headers={"X-API-Key": "chk_" + "a" * 43},
                        json={"input": "hello", "labels": ["a", "b"]},
                    )
                    self.assertEqual(r503.status_code, 503)
                    self.assertIn("error", r503.json())

                    r503_models = await pub_cli.get(
                        "/v1/models",
                        headers={"X-API-Key": "chk_" + "a" * 43},
                    )
                    self.assertEqual(r503_models.status_code, 503)
                    self.assertIn("error", r503_models.json())


if __name__ == "__main__":
    unittest.main()
