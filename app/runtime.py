"""Thread-safe mutable runtime settings shared between the public API and internal Admin API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from threading import Lock
from typing import Any, Protocol


APP_VERSION = "0.3.0"
MODEL_ID = "amber-150m"
MODEL_OWNED_BY = "MLukauskas"
SUPPORTED_MODELS = frozenset({MODEL_ID, f"{MODEL_OWNED_BY}/{MODEL_ID}"})
MAIN_PORT = 8000

DEFAULT_IDLE_TIMEOUT_SECONDS = 180
MIN_IDLE_TIMEOUT_SECONDS = 0
MAX_IDLE_TIMEOUT_SECONDS = 86400

DEFAULT_MAX_TEXT_CHARS = 8000
MIN_MAX_TEXT_CHARS = 100
MAX_MAX_TEXT_CHARS = 16000

DEFAULT_MAX_LABELS = 16
MIN_MAX_LABELS = 2
MAX_MAX_LABELS = 16

SETTINGS_SCHEMA: tuple[dict[str, Any], ...] = (
    {
        "key": "accept_requests",
        "label": "Accept classification requests",
        "type": "bool",
        "help": "When disabled, POST /v1/classify returns HTTP 503 while /healthz, /v1/models, and the Admin API remain available.",
    },
    {
        "key": "idle_timeout_seconds",
        "label": "Worker idle timeout (seconds)",
        "type": "int",
        "min": MIN_IDLE_TIMEOUT_SECONDS,
        "max": MAX_IDLE_TIMEOUT_SECONDS,
        "help": "Seconds of inactivity before unloading the model worker and releasing GPU VRAM. Set to 0 to keep the worker resident permanently.",
    },
    {
        "key": "max_text_chars",
        "label": "Max serialized text length (chars)",
        "type": "int",
        "min": MIN_MAX_TEXT_CHARS,
        "max": MAX_MAX_TEXT_CHARS,
        "help": "Maximum character length of each serialized input item accepted by POST /v1/classify (100–16,000).",
    },
    {
        "key": "max_labels",
        "label": "Max candidate labels",
        "type": "int",
        "min": MIN_MAX_LABELS,
        "max": MAX_MAX_LABELS,
        "help": "Maximum number of competing candidate labels accepted per POST /v1/classify request (2–16).",
    },
)

_INT_BOUNDS: dict[str, tuple[int, int]] = {
    "idle_timeout_seconds": (MIN_IDLE_TIMEOUT_SECONDS, MAX_IDLE_TIMEOUT_SECONDS),
    "max_text_chars": (MIN_MAX_TEXT_CHARS, MAX_MAX_TEXT_CHARS),
    "max_labels": (MIN_MAX_LABELS, MAX_MAX_LABELS),
}
_ALLOWED_KEYS = frozenset({"accept_requests", *_INT_BOUNDS})


class SettingsValidationError(ValueError):
    """Raised when an Admin API settings patch payload fails validation."""


class WorkerController(Protocol):
    """Subset of ModelWorker controlled and shared via ChoiceRuntime."""

    def set_idle_timeout(self, idle_timeout: float) -> None: ...

    def classify_batch(
        self,
        items: Sequence[tuple[str, list[str] | dict[str, str | None], str, str | None]],
    ) -> list[tuple[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class SettingsSnapshot:
    """Consistent point-in-time view of mutable runtime settings."""

    accept_requests: bool
    idle_timeout_seconds: int
    max_text_chars: int
    max_labels: int

    def to_dict(self) -> dict[str, bool | int]:
        return {
            "accept_requests": self.accept_requests,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "max_text_chars": self.max_text_chars,
            "max_labels": self.max_labels,
        }


def _validate_int_range(key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SettingsValidationError(f"{key} must be an integer")
    low, high = _INT_BOUNDS[key]
    if value < low or value > high:
        raise SettingsValidationError(f"{key} must be between {low} and {high}")
    return value


class ChoiceRuntime:
    """Coordinates live settings and worker lifecycle safely across servers and threads."""

    def __init__(
        self,
        *,
        accept_requests: bool = True,
        idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS,
        worker_idle_timeout: float | None = None,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
        max_labels: int = DEFAULT_MAX_LABELS,
    ) -> None:
        if not isinstance(accept_requests, bool):
            raise ValueError("accept_requests must be a boolean")
        _validate_int_range("idle_timeout_seconds", idle_timeout_seconds)
        _validate_int_range("max_text_chars", max_text_chars)
        _validate_int_range("max_labels", max_labels)

        raw_worker_timeout = float(idle_timeout_seconds if worker_idle_timeout is None else worker_idle_timeout)
        if not math.isfinite(raw_worker_timeout) or raw_worker_timeout < 0 or raw_worker_timeout > MAX_IDLE_TIMEOUT_SECONDS:
            raise ValueError(
                f"IDLE_TIMEOUT_SECONDS must be between {MIN_IDLE_TIMEOUT_SECONDS} and {MAX_IDLE_TIMEOUT_SECONDS}."
            )

        self._state_lock = Lock()
        self._worker_timeout_lock = Lock()
        self._accept_requests = accept_requests
        self._idle_timeout_seconds = idle_timeout_seconds
        self._worker_idle_timeout = raw_worker_timeout
        self._idle_timeout_patched = False
        self._max_text_chars = max_text_chars
        self._max_labels = max_labels
        self._worker: WorkerController | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ChoiceRuntime:
        import os

        env = os.environ if environ is None else environ
        raw_idle = env.get("IDLE_TIMEOUT_SECONDS", str(DEFAULT_IDLE_TIMEOUT_SECONDS)).strip()
        if not raw_idle:
            raw_idle = str(DEFAULT_IDLE_TIMEOUT_SECONDS)
        try:
            worker_idle = float(raw_idle)
        except ValueError as exc:
            raise ValueError(f"Invalid IDLE_TIMEOUT_SECONDS {raw_idle!r}.") from exc
        if not math.isfinite(worker_idle) or worker_idle < MIN_IDLE_TIMEOUT_SECONDS or worker_idle > MAX_IDLE_TIMEOUT_SECONDS:
            raise ValueError(
                f"IDLE_TIMEOUT_SECONDS must be between {MIN_IDLE_TIMEOUT_SECONDS} and {MAX_IDLE_TIMEOUT_SECONDS}."
            )
        # Preserve sub-second timeouts for ModelWorker if configured via env, while reporting an int >= 1 in settings.
        if 0 < worker_idle < 1:
            admin_idle = 1
        else:
            admin_idle = int(round(worker_idle))
        return cls(idle_timeout_seconds=admin_idle, worker_idle_timeout=worker_idle)

    def initial_worker_idle_timeout(self) -> float:
        with self._state_lock:
            if self._idle_timeout_patched:
                return float(self._idle_timeout_seconds)
            return self._worker_idle_timeout

    def attach_worker(self, worker: WorkerController) -> None:
        with self._worker_timeout_lock:
            with self._state_lock:
                self._worker = worker
                patched = self._idle_timeout_patched
                target_idle = float(self._idle_timeout_seconds)
            if patched:
                worker.set_idle_timeout(target_idle)

    def detach_worker(self) -> None:
        with self._worker_timeout_lock:
            with self._state_lock:
                self._worker = None

    def get_worker(self) -> WorkerController | None:
        """Return the currently attached shared ModelWorker, or None if unavailable."""
        with self._state_lock:
            return self._worker

    def snapshot(self) -> SettingsSnapshot:
        with self._state_lock:
            return SettingsSnapshot(
                accept_requests=self._accept_requests,
                idle_timeout_seconds=self._idle_timeout_seconds,
                max_text_chars=self._max_text_chars,
                max_labels=self._max_labels,
            )

    def to_dict(self) -> dict[str, bool | int]:
        return self.snapshot().to_dict()

    def apply_patch(self, patch: object) -> dict[str, bool | int]:
        if not isinstance(patch, dict):
            raise SettingsValidationError("settings patch must be a JSON object")

        validated: dict[str, bool | int] = {}
        for key, value in patch.items():
            if not isinstance(key, str) or key not in _ALLOWED_KEYS:
                raise SettingsValidationError(f"unknown setting {key!r}")
            if key == "accept_requests":
                if not isinstance(value, bool):
                    raise SettingsValidationError("accept_requests must be a boolean")
                validated[key] = value
            else:
                validated[key] = _validate_int_range(key, value)

        with self._state_lock:
            if "accept_requests" in validated:
                self._accept_requests = bool(validated["accept_requests"])
            if "idle_timeout_seconds" in validated:
                self._idle_timeout_seconds = int(validated["idle_timeout_seconds"])
                self._idle_timeout_patched = True
            if "max_text_chars" in validated:
                self._max_text_chars = int(validated["max_text_chars"])
            if "max_labels" in validated:
                self._max_labels = int(validated["max_labels"])

            result = SettingsSnapshot(
                accept_requests=self._accept_requests,
                idle_timeout_seconds=self._idle_timeout_seconds,
                max_text_chars=self._max_text_chars,
                max_labels=self._max_labels,
            ).to_dict()

        if "idle_timeout_seconds" in validated:
            with self._worker_timeout_lock:
                with self._state_lock:
                    worker = self._worker
                    latest_idle = float(self._idle_timeout_seconds)
                if worker is not None:
                    worker.set_idle_timeout(latest_idle)

        return result
