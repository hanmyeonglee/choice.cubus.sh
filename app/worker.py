"""Manage an on-demand inference subprocess that releases its CUDA context when idle."""

import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import signal
from threading import Lock, Timer
import time

_PRECISIONS = {"fp16", "float16", "fp32", "float32"}


class WorkerInputTooLongError(ValueError):
    """Raised when the worker rejects an input that exceeds the token budget."""


def validate_runtime_config(
    model_path: str,
    device: str,
    precision: str | None = None,
    idle_timeout: float = 180.0,
) -> None:
    if not Path(model_path).is_dir():
        raise RuntimeError(f"Model directory does not exist: {model_path}")
    if precision is not None and precision.strip():
        key = precision.strip().lower()
        if key not in _PRECISIONS:
            raise ValueError(f"Unsupported PRECISION {precision!r}; expected 'fp16' or 'fp32'.")
        if device == "cpu" and key in {"fp16", "float16"}:
            raise ValueError("CPU execution requires PRECISION=fp32.")
    if idle_timeout < 0:
        raise ValueError("IDLE_TIMEOUT_SECONDS must be 0 or greater.")


def _worker_main(
    conn: Connection,
    model_path: str,
    device: str,
    precision: str | None,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import torch
        from .model import InputTooLongError, classify_text, load_model, resolve_dtype

        dtype = resolve_dtype(device, precision)
        model = load_model(model_path, device, precision)
        conn.send(("ready", f"device={device} dtype={dtype} torch={torch.__version__}"))
    except BaseException as exc:
        try:
            conn.send(("startup_error", f"{type(exc).__name__}: {exc}"))
        except OSError:
            pass
        finally:
            conn.close()
        return

    try:
        with torch.inference_mode():
            while True:
                try:
                    message = conn.recv()
                except EOFError:
                    break
                if message is None:
                    break
                text, labels, task, prompt = message
                try:
                    result = classify_text(
                        model,
                        text=text,
                        labels=labels,
                        task=task,
                        prompt=prompt,
                    )
                    conn.send(("ok", result))
                except InputTooLongError as exc:
                    conn.send(("input_too_long", str(exc)))
                except Exception as exc:
                    conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


class ModelWorker:
    """Serialize inference through a child process and terminate it after idle timeout."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        precision: str | None = None,
        idle_timeout: float = 180.0,
    ) -> None:
        validate_runtime_config(model_path, device, precision, idle_timeout)
        self.model_path = model_path
        self.device = device
        self.precision = precision
        self.idle_timeout = float(idle_timeout)
        self._ctx = mp.get_context("spawn")
        self._lock = Lock()
        self._proc: mp.Process | None = None
        self._conn: Connection | None = None
        self._timer: Timer | None = None
        self._last_used = 0.0

    def is_loaded(self) -> bool:
        proc = self._proc
        return proc is not None and proc.is_alive()

    def start(self) -> None:
        with self._lock:
            self._cancel_timer_locked()
            self._ensure_worker_locked()
            self._last_used = time.monotonic()
            self._schedule_timer_locked()

    def classify(
        self,
        text: str,
        labels: list[str] | dict[str, str | None],
        task: str = "decision",
        prompt: str | None = None,
    ) -> tuple[str, dict[str, float]]:
        with self._lock:
            self._cancel_timer_locked()
            self._ensure_worker_locked()
            assert self._conn is not None
            try:
                self._conn.send((text, labels, task, prompt))
                status, payload = self._conn.recv()
            except (EOFError, OSError) as exc:
                self._stop_worker_locked("unexpected worker exit")
                raise RuntimeError("Model worker exited unexpectedly.") from exc

            self._last_used = time.monotonic()
            self._schedule_timer_locked()

            if status == "ok":
                return payload
            if status == "input_too_long":
                raise WorkerInputTooLongError(payload)
            raise RuntimeError(payload)

    def close(self) -> None:
        with self._lock:
            self._cancel_timer_locked()
            self._stop_worker_locked("server shutdown")

    def _ensure_worker_locked(self) -> None:
        if self._proc is not None and self._proc.is_alive() and self._conn is not None:
            return
        if self._proc is not None:
            self._stop_worker_locked("stale worker cleanup")

        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, self.model_path, self.device, self.precision),
            daemon=True,
        )
        proc.start()
        child_conn.close()

        try:
            status, detail = parent_conn.recv()
        except (EOFError, OSError) as exc:
            parent_conn.close()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            proc.close()
            raise RuntimeError("Model worker failed during startup.") from exc

        if status != "ready":
            parent_conn.close()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            proc.close()
            raise RuntimeError(f"Model worker startup failed: {detail}")

        self._proc = proc
        self._conn = parent_conn
        logging.info("Model worker started: pid=%s %s", proc.pid, detail)

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule_timer_locked(self) -> None:
        if self.idle_timeout <= 0:
            return
        timer = Timer(self.idle_timeout, self._unload_if_idle)
        timer.daemon = True
        timer.start()
        self._timer = timer

    def _unload_if_idle(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            idle_for = time.monotonic() - self._last_used
            if idle_for + 0.05 < self.idle_timeout:
                self._schedule_timer_locked()
                return
            self._timer = None
            self._stop_worker_locked(f"idle for {idle_for:.0f}s")

    def _stop_worker_locked(self, reason: str) -> None:
        conn = self._conn
        proc = self._proc
        self._conn = None
        self._proc = None

        if conn is not None:
            try:
                conn.send(None)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass

        if proc is not None:
            pid = proc.pid
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            proc.close()
            logging.info("Model worker stopped (%s): pid=%s GPU context released", reason, pid)
