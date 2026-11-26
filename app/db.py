"""PostgreSQL connection pool, db-broker bootstrap, and schema migration for choice.cubus.sh."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import http.client
import json
import logging
import math
import os
import random
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
from uuid import UUID

import psycopg
from psycopg_pool import ConnectionPool

from .api_keys import (
    APIKeyAuthRecord,
    APIKeyRecord,
    APIKeyStore,
    APIKeyStoreError,
)


_logger = logging.getLogger(__name__)

DEFAULT_POOL_MIN_SIZE = 1
DEFAULT_POOL_MAX_SIZE = 8
DEFAULT_POOL_TIMEOUT_SECONDS = 10.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5
DEFAULT_STATEMENT_TIMEOUT_MS = 5000
DEFAULT_KEEPALIVES_IDLE_SECONDS = 30
DEFAULT_KEEPALIVES_INTERVAL_SECONDS = 10
DEFAULT_KEEPALIVES_COUNT = 3
MAX_BROKER_RESPONSE_BYTES = 64 << 10  # 64 KiB
MAX_RETRY_DELAY_SECONDS = 10.0
_MIGRATION_LOCK_KEY = 728391045

CREATE_API_KEYS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    owner TEXT NOT NULL DEFAULT '',
    key_prefix TEXT NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked')),
    expires_at TIMESTAMPTZ NULL,
    last_used_at TIMESTAMPTZ NULL,
    request_count BIGINT NOT NULL DEFAULT 0,
    created_by TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ NULL
);
CREATE INDEX IF NOT EXISTS idx_api_keys_created_at ON api_keys (created_at DESC);
"""


class DatabaseConfigError(RuntimeError):
    """Raised when DB_BROKER_URL or DATABASE_URL configuration cannot be resolved."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject HTTP redirects on internal db-broker requests instead of following them."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


def _is_postgres_dsn(value: str) -> bool:
    return value.startswith("postgresql://") or value.startswith("postgres://")


def _parse_retry_after(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        delay = float(raw.strip())
    except ValueError:
        return None
    if not math.isfinite(delay) or delay <= 0:
        return None
    return min(delay, MAX_RETRY_DELAY_SECONDS)


def _validate_broker_url(raw_url: str) -> str:
    parsed = urllib.parse.urlsplit(raw_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise DatabaseConfigError("DB_BROKER_URL must be an http:// or https:// URL.")
    if parsed.username or parsed.password or parsed.fragment:
        raise DatabaseConfigError("DB_BROKER_URL must not contain credentials or URL fragments.")
    return raw_url


def resolve_database_config(
    environ: Mapping[str, str] | None = None,
    *,
    max_attempts: int = 5,
    timeout: float = 10.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    jitter_fn: Callable[[float, float], float] = random.uniform,
    opener: Any | None = None,
) -> dict[str, Any]:
    """Resolve PostgreSQL connection config via DB_BROKER_URL (production) or DATABASE_URL (standalone).

    Never logs or embeds returned passwords or DSNs in logs or exception messages.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    env = os.environ if environ is None else environ
    broker_url = env.get("DB_BROKER_URL", "").strip()
    if not broker_url:
        dsn = env.get("DATABASE_URL", "").strip()
        if not dsn:
            raise DatabaseConfigError("Set DB_BROKER_URL or DATABASE_URL before starting the server.")
        if not _is_postgres_dsn(dsn):
            raise DatabaseConfigError("DATABASE_URL must start with postgresql:// or postgres://.")
        return {"url": dsn}

    broker_url = _validate_broker_url(broker_url)
    active_opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirectHandler())
        if opener is None
        else opener
    )
    req = urllib.request.Request(
        broker_url,
        data=b"{}",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    last_reason = "unknown error"
    for attempt in range(max_attempts):
        delay: float | None = None
        try:
            with active_opener.open(req, timeout=timeout) as resp:
                raw_body = resp.read(MAX_BROKER_RESPONSE_BYTES + 1)
                if len(raw_body) > MAX_BROKER_RESPONSE_BYTES:
                    raise DatabaseConfigError("db-broker response exceeded maximum size.")
                try:
                    data = json.loads(raw_body.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    raise DatabaseConfigError("db-broker returned malformed JSON.") from None
                if not isinstance(data, dict):
                    raise DatabaseConfigError("db-broker returned a non-object JSON payload.")
                raw_dsn = data.get("url")
                if not isinstance(raw_dsn, str) or not _is_postgres_dsn(raw_dsn.strip()):
                    raise DatabaseConfigError("db-broker response did not include a valid PostgreSQL URL.")
                _logger.info(
                    "Resolved PostgreSQL credentials from db-broker (database=%s, host=%s)",
                    data.get("database", "<unknown>"),
                    data.get("host", "<unknown>"),
                )
                return data
        except DatabaseConfigError:
            raise
        except urllib.error.HTTPError as exc:
            code = exc.code
            retry_after = _parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
            try:
                exc.close()
            except Exception:
                pass
            if 300 <= code < 500:
                raise DatabaseConfigError(f"db-broker rejected credential request (HTTP {code}).") from None
            last_reason = f"HTTP {code}"
            delay = retry_after if retry_after is not None else min(0.5 * (2**attempt), MAX_RETRY_DELAY_SECONDS)
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
            last_reason = type(exc).__name__
            delay = min(0.5 * (2**attempt), MAX_RETRY_DELAY_SECONDS)

        if attempt == max_attempts - 1:
            raise DatabaseConfigError(
                f"Failed to resolve PostgreSQL credentials from db-broker after {max_attempts} attempts ({last_reason})."
            ) from None

        jitter = max(0.0, float(jitter_fn(0.0, 0.25)))
        sleep_fn(delay + jitter)

    raise DatabaseConfigError("Failed to resolve PostgreSQL credentials from db-broker.")


def _row_to_record(row: tuple[Any, ...]) -> APIKeyRecord:
    return APIKeyRecord(
        id=row[0] if isinstance(row[0], UUID) else UUID(str(row[0])),
        name=str(row[1]),
        owner=str(row[2]),
        key_prefix=str(row[3]),
        status=str(row[4]),
        expires_at=row[5],
        last_used_at=row[6],
        request_count=int(row[7]),
        created_by=str(row[8]),
        created_at=row[9],
        revoked_at=row[10],
    )


class PostgresAPIKeyBackend:
    """PostgreSQL-backed APIKeyBackend using psycopg_pool.ConnectionPool."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        pool: Any | None = None,
        min_size: int = DEFAULT_POOL_MIN_SIZE,
        max_size: int = DEFAULT_POOL_MAX_SIZE,
        timeout: float = DEFAULT_POOL_TIMEOUT_SECONDS,
        connect_timeout: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
    ) -> None:
        if pool is not None:
            self._pool = pool
        else:
            if not dsn or not _is_postgres_dsn(dsn.strip()):
                raise DatabaseConfigError("A valid PostgreSQL DSN is required.")
            if min_size < 1 or max_size < min_size or max_size > 12:
                raise ValueError("Pool sizes must satisfy 1 <= min_size <= max_size <= 12.")
            if statement_timeout_ms < 100:
                raise ValueError("statement_timeout_ms must be >= 100.")
            self._pool = ConnectionPool(
                conninfo=dsn.strip(),
                min_size=min_size,
                max_size=max_size,
                timeout=timeout,
                open=True,
                check=ConnectionPool.check_connection,
                kwargs={
                    "connect_timeout": connect_timeout,
                    "options": f"-c statement_timeout={int(statement_timeout_ms)}",
                    "keepalives": 1,
                    "keepalives_idle": DEFAULT_KEEPALIVES_IDLE_SECONDS,
                    "keepalives_interval": DEFAULT_KEEPALIVES_INTERVAL_SECONDS,
                    "keepalives_count": DEFAULT_KEEPALIVES_COUNT,
                },
            )

    def initialize(self, *, wait_timeout: float = 15.0) -> None:
        """Wait for pool readiness and apply idempotent schema migration."""
        try:
            wait_fn = getattr(self._pool, "wait", None)
            if callable(wait_fn):
                wait_fn(timeout=wait_timeout)
            with self._pool.connection() as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_KEY,))
                        cur.execute(CREATE_API_KEYS_TABLE_SQL)
        except psycopg.Error as exc:
            _logger.error("PostgreSQL initialization/migration failed: %s", type(exc).__name__)
            raise APIKeyStoreError(f"Database initialization failed ({type(exc).__name__}).") from None

    def lookup_by_hash(self, key_hash: str) -> APIKeyAuthRecord | None:
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, key_hash, status, expires_at FROM api_keys WHERE key_hash = %s",
                        (key_hash,),
                    )
                    row = cur.fetchone()
        except psycopg.Error as exc:
            _logger.warning("PostgreSQL API key lookup failed: %s", type(exc).__name__)
            raise APIKeyStoreError("Database lookup failed.") from None

        if row is None:
            return None
        return APIKeyAuthRecord(
            id=row[0] if isinstance(row[0], UUID) else UUID(str(row[0])),
            key_hash=str(row[1]),
            status=str(row[2]),
            expires_at=row[3],
        )

    def list_keys(self) -> list[APIKeyRecord]:
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, name, owner, key_prefix, status, expires_at,
                               last_used_at, request_count, created_by, created_at, revoked_at
                          FROM api_keys
                         ORDER BY created_at DESC, id ASC
                        """
                    )
                    rows = cur.fetchall()
        except psycopg.Error as exc:
            _logger.warning("PostgreSQL API key list failed: %s", type(exc).__name__)
            raise APIKeyStoreError("Database list query failed.") from None

        return [_row_to_record(row) for row in rows]

    def create_key(self, record: APIKeyRecord, key_hash: str) -> APIKeyRecord:
        try:
            with self._pool.connection() as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO api_keys (
                                id, name, owner, key_prefix, key_hash, status,
                                expires_at, last_used_at, request_count, created_by, created_at, revoked_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id, name, owner, key_prefix, status, expires_at,
                                      last_used_at, request_count, created_by, created_at, revoked_at
                            """,
                            (
                                record.id,
                                record.name,
                                record.owner,
                                record.key_prefix,
                                key_hash,
                                record.status,
                                record.expires_at,
                                record.last_used_at,
                                record.request_count,
                                record.created_by,
                                record.created_at,
                                record.revoked_at,
                            ),
                        )
                        row = cur.fetchone()
        except psycopg.Error as exc:
            _logger.warning("PostgreSQL API key insert failed: %s", type(exc).__name__)
            raise APIKeyStoreError("Database insert failed.") from None

        if row is None:
            raise APIKeyStoreError("Database insert returned no row.")
        return _row_to_record(row)

    def revoke_key(self, key_id: UUID, revoked_at: datetime) -> tuple[APIKeyRecord, str] | None:
        try:
            with self._pool.connection() as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE api_keys
                               SET status = 'revoked',
                                   revoked_at = COALESCE(revoked_at, %s)
                             WHERE id = %s
                            RETURNING id, name, owner, key_prefix, status, expires_at,
                                      last_used_at, request_count, created_by, created_at, revoked_at, key_hash
                            """,
                            (revoked_at, key_id),
                        )
                        row = cur.fetchone()
        except psycopg.Error as exc:
            _logger.warning("PostgreSQL API key revoke failed: %s", type(exc).__name__)
            raise APIKeyStoreError("Database revoke failed.") from None

        if row is None:
            return None
        return _row_to_record(row[:11]), str(row[11])

    def delete_key(self, key_id: UUID) -> str | None:
        try:
            with self._pool.connection() as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM api_keys WHERE id = %s RETURNING key_hash",
                            (key_id,),
                        )
                        row = cur.fetchone()
        except psycopg.Error as exc:
            _logger.warning("PostgreSQL API key delete failed: %s", type(exc).__name__)
            raise APIKeyStoreError("Database delete failed.") from None

        if row is None:
            return None
        return str(row[0])

    def flush_usage_batch(self, updates: Sequence[tuple[UUID, int, datetime]]) -> None:
        if not updates:
            return
        params = [(count_delta, last_used_at, last_used_at, key_id) for key_id, count_delta, last_used_at in updates]
        try:
            with self._pool.connection() as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.executemany(
                            """
                            UPDATE api_keys
                               SET request_count = request_count + %s,
                                   last_used_at = GREATEST(COALESCE(last_used_at, %s), %s)
                             WHERE id = %s
                            """,
                            params,
                        )
        except psycopg.Error as exc:
            _logger.warning("PostgreSQL API key usage flush failed: %s", type(exc).__name__)
            raise APIKeyStoreError("Database usage flush failed.") from None

    def close(self) -> None:
        try:
            self._pool.close()
        except Exception as exc:
            _logger.warning("Error closing PostgreSQL pool: %s", type(exc).__name__)


def create_postgres_api_key_store(environ: Mapping[str, str] | None = None) -> APIKeyStore:
    """Resolve database credentials, initialize schema, and construct an APIKeyStore."""
    config = resolve_database_config(environ)
    backend = PostgresAPIKeyBackend(config["url"])
    try:
        backend.initialize()
    except BaseException:
        backend.close()
        raise
    return APIKeyStore(backend)
