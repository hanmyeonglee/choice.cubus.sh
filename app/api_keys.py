"""DB-backed API key management, fast bounded-TTL auth cache, and async batched usage tracking."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import logging
import secrets
import string
from threading import Event, Lock, Thread
import time
from typing import Any, Literal, Protocol
import unicodedata
from uuid import UUID, uuid4


API_KEY_PREFIX_TAG = "chk_"
API_KEY_PREFIX_LEN = 12
MIN_API_KEY_LEN = 16
MAX_API_KEY_LEN = 128

DEFAULT_CACHE_TTL_SECONDS = 15.0
DEFAULT_CACHE_MAX_SIZE = 1024
DEFAULT_FLUSH_INTERVAL_SECONDS = 5.0
MAX_PENDING_USAGE_KEYS = 4096

ALLOWED_EXPIRES_IN_DAYS = frozenset({30, 90, 365})
MAX_KEY_NAME_CHARS = 120
MAX_KEY_OWNER_CHARS = 120
MAX_KEY_CREATED_BY_CHARS = 120

_VALID_KEY_CHARS = frozenset(string.ascii_letters + string.digits + "-_")
_logger = logging.getLogger(__name__)


class APIKeyStoreError(RuntimeError):
    """Raised when the backing API key persistence store is temporarily unavailable."""


class APIKeyValidationError(ValueError):
    """Raised when an API key management payload or identifier fails validation."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_iso_utc(dt: datetime | None) -> str | None:
    norm = _normalize_utc(dt)
    if norm is None:
        return None
    return norm.isoformat().replace("+00:00", "Z")


def hash_api_key(raw_key: str) -> str:
    """Return the lowercase SHA-256 hex digest of a raw API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def is_valid_key_format(raw_key: str) -> bool:
    """Fast syntactic pre-check rejecting malformed keys before hashing or DB lookup."""
    if not isinstance(raw_key, str):
        return False
    if len(raw_key) < MIN_API_KEY_LEN or len(raw_key) > MAX_API_KEY_LEN:
        return False
    if not raw_key.startswith(API_KEY_PREFIX_TAG):
        return False
    body = raw_key[len(API_KEY_PREFIX_TAG) :]
    return bool(body) and all(ch in _VALID_KEY_CHARS for ch in body)


def generate_api_key() -> tuple[str, str, str]:
    """Generate a fresh API key returning (plaintext, key_prefix, key_hash)."""
    raw_key = f"{API_KEY_PREFIX_TAG}{secrets.token_urlsafe(32)}"
    return raw_key, raw_key[:API_KEY_PREFIX_LEN], hash_api_key(raw_key)


def parse_api_key_uuid(raw_id: str) -> UUID:
    """Parse and validate a canonical hyphenated UUID string."""
    if not isinstance(raw_id, str) or len(raw_id) != 36:
        raise APIKeyValidationError("invalid api key id")
    try:
        parsed = UUID(raw_id)
    except ValueError as exc:
        raise APIKeyValidationError("invalid api key id") from exc
    if str(parsed) != raw_id.lower():
        raise APIKeyValidationError("invalid api key id")
    return parsed


def _validate_metadata_string(value: object, field: str, max_len: int, *, required: bool) -> str:
    if not isinstance(value, str):
        raise APIKeyValidationError(f"{field} must be a string")
    if any(unicodedata.category(ch) == "Cc" for ch in value):
        raise APIKeyValidationError(f"{field} must not contain control characters")
    stripped = value.strip()
    if required and not stripped:
        raise APIKeyValidationError(f"{field} must not be blank")
    if len(stripped) > max_len:
        raise APIKeyValidationError(f"{field} must not exceed {max_len} characters")
    return stripped


def validate_create_api_key_payload(parsed: dict[str, Any]) -> tuple[str, str, int | None, str]:
    """Strictly validate POST /admin/v1/api-keys request body."""
    allowed = {"name", "owner", "expires_in_days", "created_by"}
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise APIKeyValidationError(f"unknown field {unknown[0]!r}")
    if "name" not in parsed:
        raise APIKeyValidationError("name is required")

    name = _validate_metadata_string(parsed["name"], "name", MAX_KEY_NAME_CHARS, required=True)
    owner = (
        _validate_metadata_string(parsed["owner"], "owner", MAX_KEY_OWNER_CHARS, required=False)
        if "owner" in parsed
        else ""
    )
    created_by = (
        _validate_metadata_string(parsed["created_by"], "created_by", MAX_KEY_CREATED_BY_CHARS, required=False)
        if "created_by" in parsed
        else ""
    )

    expires_in_days: int | None = None
    if "expires_in_days" in parsed and parsed["expires_in_days"] is not None:
        raw_days = parsed["expires_in_days"]
        if isinstance(raw_days, bool) or not isinstance(raw_days, int) or raw_days not in ALLOWED_EXPIRES_IN_DAYS:
            raise APIKeyValidationError("expires_in_days must be one of 30, 90, 365, or null")
        expires_in_days = raw_days

    return name, owner, expires_in_days, created_by


@dataclass(frozen=True, slots=True)
class APIKeyRecord:
    """Public metadata for an API key (never holds plaintext or key_hash)."""

    id: UUID
    name: str
    owner: str
    key_prefix: str
    status: str
    expires_at: datetime | None
    last_used_at: datetime | None
    request_count: int
    created_by: str
    created_at: datetime
    revoked_at: datetime | None

    def effective_status(self, now: datetime) -> str:
        if self.status == "revoked":
            return "revoked"
        exp = _normalize_utc(self.expires_at)
        if exp is not None and exp <= _normalize_utc(now):  # type: ignore[operator]
            return "expired"
        return "active"

    def to_dict(
        self,
        now: datetime,
        *,
        pending_count: int = 0,
        pending_last_used_at: datetime | None = None,
    ) -> dict[str, Any]:
        last_used = _normalize_utc(self.last_used_at)
        pending_ts = _normalize_utc(pending_last_used_at)
        if pending_ts is not None and (last_used is None or pending_ts > last_used):
            last_used = pending_ts

        return {
            "id": str(self.id),
            "name": self.name,
            "owner": self.owner,
            "key_prefix": self.key_prefix,
            "status": self.effective_status(now),
            "expires_at": _format_iso_utc(self.expires_at),
            "last_used_at": _format_iso_utc(last_used),
            "request_count": self.request_count + pending_count,
            "created_by": self.created_by,
            "created_at": _format_iso_utc(self.created_at),
            "revoked_at": _format_iso_utc(self.revoked_at),
        }


@dataclass(frozen=True, slots=True)
class APIKeyAuthRecord:
    """Minimal credential verification projection loaded by key_hash."""

    id: UUID
    key_hash: str
    status: str
    expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    id: UUID
    key_hash: str
    status: str
    expires_at: datetime | None
    cached_at: float


class APIKeyBackend(Protocol):
    """Persistence interface implemented by PostgresAPIKeyBackend and in-memory fakes."""

    def lookup_by_hash(self, key_hash: str) -> APIKeyAuthRecord | None: ...

    def list_keys(self) -> list[APIKeyRecord]: ...

    def create_key(self, record: APIKeyRecord, key_hash: str) -> APIKeyRecord: ...

    def revoke_key(self, key_id: UUID, revoked_at: datetime) -> tuple[APIKeyRecord, str] | None: ...

    def delete_key(self, key_id: UUID) -> str | None: ...

    def flush_usage_batch(self, updates: Sequence[tuple[UUID, int, datetime]]) -> None: ...

    def close(self) -> None: ...


class InMemoryAPIKeyBackend:
    """Thread-safe in-memory APIKeyBackend for fast unit and integration tests."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: dict[UUID, tuple[APIKeyRecord, str]] = {}
        self._hash_to_id: dict[str, UUID] = {}
        self.lookup_calls = 0
        self.flush_calls = 0
        self.closed = False

    def lookup_by_hash(self, key_hash: str) -> APIKeyAuthRecord | None:
        with self._lock:
            self.lookup_calls += 1
            key_id = self._hash_to_id.get(key_hash)
            if key_id is None:
                return None
            rec, stored_hash = self._records[key_id]
            return APIKeyAuthRecord(
                id=rec.id,
                key_hash=stored_hash,
                status=rec.status,
                expires_at=rec.expires_at,
            )

    def list_keys(self) -> list[APIKeyRecord]:
        with self._lock:
            items = [rec for rec, _ in self._records.values()]
        items.sort(key=lambda r: (-_normalize_utc(r.created_at).timestamp(), str(r.id)))  # type: ignore[union-attr]
        return items

    def create_key(self, record: APIKeyRecord, key_hash: str) -> APIKeyRecord:
        with self._lock:
            if key_hash in self._hash_to_id:
                raise APIKeyStoreError("Duplicate API key hash.")
            self._records[record.id] = (record, key_hash)
            self._hash_to_id[key_hash] = record.id
            return record

    def revoke_key(self, key_id: UUID, revoked_at: datetime) -> tuple[APIKeyRecord, str] | None:
        with self._lock:
            entry = self._records.get(key_id)
            if entry is None:
                return None
            rec, key_hash = entry
            effective_revoked_at = rec.revoked_at if rec.revoked_at is not None else revoked_at
            updated = APIKeyRecord(
                id=rec.id,
                name=rec.name,
                owner=rec.owner,
                key_prefix=rec.key_prefix,
                status="revoked",
                expires_at=rec.expires_at,
                last_used_at=rec.last_used_at,
                request_count=rec.request_count,
                created_by=rec.created_by,
                created_at=rec.created_at,
                revoked_at=effective_revoked_at,
            )
            self._records[key_id] = (updated, key_hash)
            return updated, key_hash

    def delete_key(self, key_id: UUID) -> str | None:
        with self._lock:
            entry = self._records.pop(key_id, None)
            if entry is None:
                return None
            _, key_hash = entry
            self._hash_to_id.pop(key_hash, None)
            return key_hash

    def flush_usage_batch(self, updates: Sequence[tuple[UUID, int, datetime]]) -> None:
        with self._lock:
            self.flush_calls += 1
            for key_id, count_delta, last_used_at in updates:
                entry = self._records.get(key_id)
                if entry is None:
                    continue
                rec, key_hash = entry
                prev_ts = _normalize_utc(rec.last_used_at)
                new_ts = _normalize_utc(last_used_at)
                merged_ts = new_ts if (prev_ts is None or (new_ts is not None and new_ts > prev_ts)) else prev_ts
                updated = APIKeyRecord(
                    id=rec.id,
                    name=rec.name,
                    owner=rec.owner,
                    key_prefix=rec.key_prefix,
                    status=rec.status,
                    expires_at=rec.expires_at,
                    last_used_at=merged_ts,
                    request_count=rec.request_count + count_delta,
                    created_by=rec.created_by,
                    created_at=rec.created_at,
                    revoked_at=rec.revoked_at,
                )
                self._records[key_id] = (updated, key_hash)

    def close(self) -> None:
        with self._lock:
            self.closed = True


class APIKeyStore:
    """High-performance API key authenticator and manager backed by an APIKeyBackend."""

    def __init__(
        self,
        backend: APIKeyBackend,
        *,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        cache_max_size: int = DEFAULT_CACHE_MAX_SIZE,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        now_fn: Callable[[], datetime] = _utc_now,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be >= 0")
        if cache_max_size < 1:
            raise ValueError("cache_max_size must be >= 1")
        if flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be > 0")

        self._backend = backend
        self._cache_ttl = float(cache_ttl_seconds)
        self._cache_max_size = int(cache_max_size)
        self._flush_interval = float(flush_interval_seconds)
        self._now_fn = now_fn
        self._mono_fn = monotonic_fn

        self._cache_lock = Lock()
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._id_to_hash: dict[UUID, str] = {}
        self._invalidation_gen = 0

        self._usage_lock = Lock()
        self._pending_usage: dict[UUID, tuple[int, datetime]] = {}
        self._inflight_usage: dict[UUID, tuple[int, datetime]] = {}
        self._flush_lock = Lock()

        self._flusher_lock = Lock()
        self._stop_event = Event()
        self._flusher_thread: Thread | None = None
        self._shutdown_flushed = False
        self._closed = False

    def _now_utc(self) -> datetime:
        return _normalize_utc(self._now_fn())  # type: ignore[return-value]

    def _evict_locked(self, *, key_id: UUID | None = None, key_hash: str | None = None) -> None:
        if key_hash is not None:
            entry = self._cache.pop(key_hash, None)
            if entry is not None and self._id_to_hash.get(entry.id) == key_hash:
                self._id_to_hash.pop(entry.id, None)
        if key_id is not None:
            mapped_hash = self._id_to_hash.pop(key_id, None)
            if mapped_hash is not None:
                self._cache.pop(mapped_hash, None)

    def _record_usage(self, key_id: UUID, used_at: datetime) -> None:
        with self._usage_lock:
            self._shutdown_flushed = False
            existing = self._pending_usage.get(key_id)
            if existing is None:
                if len(self._pending_usage) < MAX_PENDING_USAGE_KEYS:
                    self._pending_usage[key_id] = (1, used_at)
            else:
                prev_count, prev_ts = existing
                latest_ts = used_at if used_at > prev_ts else prev_ts
                self._pending_usage[key_id] = (prev_count + 1, latest_ts)

    @staticmethod
    def _merge_usage_pair(
        first: tuple[int, datetime] | None,
        second: tuple[int, datetime] | None,
    ) -> tuple[int, datetime | None]:
        if first is None and second is None:
            return 0, None
        if first is None:
            assert second is not None
            return second[0], second[1]
        if second is None:
            return first[0], first[1]
        return first[0] + second[0], first[1] if first[1] >= second[1] else second[1]

    def _get_pending_usage(self, key_id: UUID) -> tuple[int, datetime | None]:
        with self._usage_lock:
            return self._merge_usage_pair(
                self._pending_usage.get(key_id),
                self._inflight_usage.get(key_id),
            )

    def _snapshot_pending_usage(self) -> dict[UUID, tuple[int, datetime]]:
        with self._usage_lock:
            merged: dict[UUID, tuple[int, datetime]] = dict(self._pending_usage)
            for key_id, (count, ts) in self._inflight_usage.items():
                existing = merged.get(key_id)
                if existing is None:
                    merged[key_id] = (count, ts)
                else:
                    prev_count, prev_ts = existing
                    merged[key_id] = (prev_count + count, ts if ts > prev_ts else prev_ts)
            return merged

    def verify_key_fast(self, raw_key: str) -> tuple[Literal["hit", "invalid", "miss"], Any]:
        """Non-blocking cache-first verification step safe to call directly on the event loop."""
        if not is_valid_key_format(raw_key):
            return ("invalid", None)

        key_hash = hash_api_key(raw_key)
        now_mono = self._mono_fn()
        now_utc = self._now_utc()

        with self._cache_lock:
            entry = self._cache.get(key_hash)
            if entry is not None:
                if self._cache_ttl > 0 and (now_mono - entry.cached_at) < self._cache_ttl:
                    self._cache.move_to_end(key_hash)
                    exp = _normalize_utc(entry.expires_at)
                    if (
                        entry.status == "active"
                        and hmac.compare_digest(entry.key_hash, key_hash)
                        and (exp is None or exp > now_utc)
                    ):
                        key_id = entry.id
                    else:
                        self._evict_locked(key_id=entry.id, key_hash=key_hash)
                        return ("invalid", None)
                else:
                    self._evict_locked(key_id=entry.id, key_hash=key_hash)
                    key_id = None
            else:
                key_id = None

            if key_id is None:
                return ("miss", (key_hash, self._invalidation_gen))

        self._record_usage(key_id, now_utc)
        return ("hit", key_id)

    def verify_key_miss(self, key_hash: str, gen_before: int) -> UUID | None:
        """Blocking DB lookup path for cache misses (safe to run in a worker thread)."""
        current_gen = gen_before
        for attempt in range(3):
            auth_rec = self._backend.lookup_by_hash(key_hash)
            if auth_rec is None:
                return None
            if not hmac.compare_digest(auth_rec.key_hash, key_hash):
                return None

            now_utc = self._now_utc()
            exp = _normalize_utc(auth_rec.expires_at)
            if auth_rec.status != "active" or (exp is not None and exp <= now_utc):
                with self._cache_lock:
                    self._evict_locked(key_id=auth_rec.id, key_hash=key_hash)
                return None

            with self._cache_lock:
                if self._invalidation_gen != current_gen:
                    current_gen = self._invalidation_gen
                    if attempt < 2:
                        continue
                    # Concurrent revocations occurred during lookup; skip caching this entry.
                    break
                if self._cache_ttl > 0:
                    entry = _CacheEntry(
                        id=auth_rec.id,
                        key_hash=auth_rec.key_hash,
                        status=auth_rec.status,
                        expires_at=exp,
                        cached_at=self._mono_fn(),
                    )
                    self._cache[key_hash] = entry
                    self._cache.move_to_end(key_hash)
                    self._id_to_hash[auth_rec.id] = key_hash
                    while len(self._cache) > self._cache_max_size:
                        evicted_hash, evicted_entry = self._cache.popitem(last=False)
                        if self._id_to_hash.get(evicted_entry.id) == evicted_hash:
                            self._id_to_hash.pop(evicted_entry.id, None)
                break

        self._record_usage(auth_rec.id, now_utc)
        return auth_rec.id

    def verify_key(self, raw_key: str) -> UUID | None:
        """Synchronous verification helper combining fast cache lookup and DB fallback."""
        state, payload = self.verify_key_fast(raw_key)
        if state == "hit":
            return payload
        if state == "invalid":
            return None
        key_hash, gen_before = payload
        return self.verify_key_miss(key_hash, gen_before)

    def list_keys(self) -> list[dict[str, Any]]:
        """List all API key records with effective status and live pending usage overlaid."""
        records = self._backend.list_keys()
        pending = self._snapshot_pending_usage()
        now_utc = self._now_utc()
        result: list[dict[str, Any]] = []
        for rec in records:
            p_count, p_ts = pending.get(rec.id, (0, None))
            result.append(rec.to_dict(now_utc, pending_count=p_count, pending_last_used_at=p_ts))
        return result

    def create_key(
        self,
        *,
        name: str,
        owner: str = "",
        expires_in_days: int | None = None,
        created_by: str = "",
    ) -> tuple[str, dict[str, Any]]:
        """Create a new API key in persistence and return (plaintext_once, record_dict)."""
        v_name, v_owner, v_days, v_created_by = validate_create_api_key_payload(
            {
                "name": name,
                "owner": owner,
                "expires_in_days": expires_in_days,
                "created_by": created_by,
            }
        )
        now_utc = self._now_utc()
        expires_at = (now_utc + timedelta(days=v_days)) if v_days is not None else None
        raw_key, key_prefix, key_hash = generate_api_key()
        record = APIKeyRecord(
            id=uuid4(),
            name=v_name,
            owner=v_owner,
            key_prefix=key_prefix,
            status="active",
            expires_at=expires_at,
            last_used_at=None,
            request_count=0,
            created_by=v_created_by,
            created_at=now_utc,
            revoked_at=None,
        )
        stored = self._backend.create_key(record, key_hash)
        return raw_key, stored.to_dict(now_utc)

    def revoke_key(self, key_id: UUID) -> dict[str, Any] | None:
        """Revoke an API key by ID and immediately invalidate any cached auth entry."""
        now_utc = self._now_utc()
        result = self._backend.revoke_key(key_id, revoked_at=now_utc)
        if result is None:
            return None
        updated, key_hash = result
        with self._cache_lock:
            self._invalidation_gen += 1
            self._evict_locked(key_id=key_id, key_hash=key_hash)
        p_count, p_ts = self._get_pending_usage(key_id)
        return updated.to_dict(now_utc, pending_count=p_count, pending_last_used_at=p_ts)

    def delete_key(self, key_id: UUID) -> bool:
        """Delete an API key by ID and immediately invalidate cache and pending usage."""
        key_hash = self._backend.delete_key(key_id)
        if key_hash is None:
            return False
        with self._cache_lock:
            self._invalidation_gen += 1
            self._evict_locked(key_id=key_id, key_hash=key_hash)
        with self._usage_lock:
            self._pending_usage.pop(key_id, None)
            self._inflight_usage.pop(key_id, None)
        return True

    def flush_usage(self) -> int:
        """Flush buffered in-memory usage increments to the backend."""
        with self._flush_lock:
            with self._usage_lock:
                if not self._pending_usage:
                    return 0
                batch = self._pending_usage
                self._pending_usage = {}
                self._inflight_usage = dict(batch)

            updates = sorted(
                ((kid, count, ts) for kid, (count, ts) in batch.items()),
                key=lambda item: str(item[0]),
            )
            try:
                self._backend.flush_usage_batch(updates)
                with self._usage_lock:
                    self._inflight_usage.clear()
                return len(updates)
            except Exception as exc:
                _logger.warning(
                    "Failed to flush API key usage batch (%d keys): %s",
                    len(updates),
                    type(exc).__name__,
                )
                with self._usage_lock:
                    self._inflight_usage.clear()
                    for kid, count, ts in updates:
                        existing = self._pending_usage.get(kid)
                        if existing is None:
                            if len(self._pending_usage) < MAX_PENDING_USAGE_KEYS:
                                self._pending_usage[kid] = (count, ts)
                        else:
                            cur_count, cur_ts = existing
                            latest_ts = ts if ts > cur_ts else cur_ts
                            self._pending_usage[kid] = (cur_count + count, latest_ts)
                return 0

    def _flush_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._flush_interval):
            self.flush_usage()

    def start_background_flush(self) -> None:
        """Start the periodic background usage flusher thread (idempotent)."""
        with self._flusher_lock:
            if self._closed:
                return
            if self._flusher_thread is not None and self._flusher_thread.is_alive():
                return
            self._stop_event.clear()
            with self._usage_lock:
                self._shutdown_flushed = False
            thread = Thread(target=self._flush_loop, name="choice-apikey-usage-flush", daemon=True)
            self._flusher_thread = thread
            thread.start()

    def stop_background_flush(self, *, flush: bool = True) -> None:
        """Stop the periodic flusher thread and optionally flush remaining buffered usage."""
        with self._flusher_lock:
            thread = self._flusher_thread
            self._stop_event.set()
            self._flusher_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        if flush:
            self.flush_usage()
            with self._usage_lock:
                self._shutdown_flushed = True

    def close(self) -> None:
        """Stop flusher, perform final usage flush, clear cache, and close backend (idempotent)."""
        with self._flusher_lock:
            if self._closed:
                return
            self._closed = True
            thread = self._flusher_thread
            self._stop_event.set()
            self._flusher_thread = None

        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

        try:
            with self._usage_lock:
                already_flushed = self._shutdown_flushed
            if not already_flushed:
                self.flush_usage()
        finally:
            with self._cache_lock:
                self._invalidation_gen += 1
                self._cache.clear()
                self._id_to_hash.clear()
            self._backend.close()
