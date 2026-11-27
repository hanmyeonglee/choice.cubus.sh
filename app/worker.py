"""Manage an on-demand micro-batching inference subprocess that releases its CUDA context when idle."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import logging
import math
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import signal
from threading import Condition, Event, Lock, Thread, Timer, current_thread, main_thread
import time
from typing import Any

_PRECISIONS = {"fp16", "float16", "fp32", "float32"}
_VALID_PLANES = frozenset({"interactive", "admin"})
_STARTUP_POLL_INTERVAL = 0.2
_CANCEL_WAIT_POLL_INTERVAL = 0.02


def threading_current_is_main() -> bool:
    return current_thread() is main_thread()


class WorkerInputTooLongError(ValueError):
    """Raised when the worker rejects an input that exceeds the token budget."""


class WorkerCancelledError(RuntimeError):
    """Raised when a classification request is cancelled or times out before completion."""


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
        if device.strip().lower() == "cpu" and key in {"fp16", "float16"}:
            raise ValueError("CPU execution requires PRECISION=fp32.")
    if not math.isfinite(idle_timeout) or idle_timeout < 0:
        raise ValueError("IDLE_TIMEOUT_SECONDS must be 0 or greater.")


def _worker_main(
    conn: Connection,
    model_path: str,
    device: str,
    precision: str | None,
) -> None:
    try:
        if threading_current_is_main():
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        import torch
        from .model import classify_batch, load_model, resolve_dtype

        dtype = resolve_dtype(device, precision)
        bundle = load_model(model_path, device, precision)
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
                if isinstance(message, tuple) and len(message) == 2 and message[0] == "classify_batch":
                    batch_items = message[1]
                elif isinstance(message, tuple) and len(message) == 4:
                    batch_items = [message]
                else:
                    try:
                        conn.send(("batch_error", "ValueError: Invalid worker IPC message."))
                    except OSError:
                        pass
                    break

                try:
                    results = classify_batch(bundle, batch_items)
                    if isinstance(message, tuple) and len(message) == 4:
                        status, payload = results[0]
                        conn.send((status, payload))
                    else:
                        conn.send(("batch_ok", results))
                except Exception as exc:
                    try:
                        conn.send(("batch_error", f"{type(exc).__name__}: {exc}"))
                    except OSError:
                        pass
                    break
    finally:
        conn.close()


class _BatchFuture:
    """Thread-safe completion tracker for one classify() or classify_batch() call."""

    def __init__(
        self,
        count: int,
        *,
        cancel_event: Event | None = None,
        deadline: float | None = None,
    ) -> None:
        self._lock = Lock()
        self._remaining = count
        self._cancel_event = cancel_event
        self._deadline = deadline
        self.results: list[tuple[str, Any] | None] = [None] * count
        self.fatal_error: BaseException | None = None
        self.cancelled = False
        self.done = Event()

    def cancel(self) -> bool:
        with self._lock:
            if self.done.is_set():
                return False
            self.cancelled = True
            self.done.set()
            return True

    def is_cancelled(self) -> bool:
        if self._cancel_event is not None and self._cancel_event.is_set():
            self.cancel()
        elif self._deadline is not None and time.monotonic() >= self._deadline:
            self.cancel()
        with self._lock:
            return self.cancelled

    def is_done_or_cancelled(self) -> bool:
        if self._cancel_event is not None and self._cancel_event.is_set():
            self.cancel()
        elif self._deadline is not None and time.monotonic() >= self._deadline:
            self.cancel()
        return self.done.is_set()

    def set_item_result(self, index: int, result: tuple[str, Any]) -> None:
        with self._lock:
            if self.done.is_set():
                return
            self.results[index] = result
            self._remaining -= 1
            if self._remaining == 0:
                self.done.set()

    def set_fatal_error(self, exc: BaseException) -> None:
        with self._lock:
            if self.done.is_set():
                return
            self.fatal_error = exc
            self.done.set()


class ModelWorker:
    """Coalesce inference through one dynamic micro-batching subprocess and evict after idle timeout."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        precision: str | None = None,
        idle_timeout: float = 180.0,
        *,
        coalesce_ms: float = 5.0,
        max_batch_size: int = 32,
        max_admin_batch_size: int = 16,
        long_text_chars: int = 4000,
        max_long_items_per_batch: int = 1,
        max_batch_chars: int = 32000,
        startup_timeout: float = 60.0,
    ) -> None:
        validate_runtime_config(model_path, device, precision, idle_timeout)
        if not math.isfinite(coalesce_ms) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be 0 or greater.")
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be an integer >= 1.")
        if (
            isinstance(max_admin_batch_size, bool)
            or not isinstance(max_admin_batch_size, int)
            or max_admin_batch_size < 1
        ):
            raise ValueError("max_admin_batch_size must be an integer >= 1.")
        if isinstance(long_text_chars, bool) or not isinstance(long_text_chars, int) or long_text_chars < 1:
            raise ValueError("long_text_chars must be an integer >= 1.")
        if (
            isinstance(max_long_items_per_batch, bool)
            or not isinstance(max_long_items_per_batch, int)
            or max_long_items_per_batch < 1
        ):
            raise ValueError("max_long_items_per_batch must be an integer >= 1.")
        if isinstance(max_batch_chars, bool) or not isinstance(max_batch_chars, int) or max_batch_chars < 1:
            raise ValueError("max_batch_chars must be an integer >= 1.")
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("startup_timeout must be greater than 0.")

        self.model_path = model_path
        self.device = device
        self.precision = precision
        self.idle_timeout = float(idle_timeout)
        self.coalesce_ms = float(coalesce_ms)
        self.max_batch_size = max_batch_size
        self.max_admin_batch_size = max_admin_batch_size
        self.long_text_chars = long_text_chars
        self.max_long_items_per_batch = max_long_items_per_batch
        self.max_batch_chars = max_batch_chars
        self.startup_timeout = float(startup_timeout)

        self._ctx = mp.get_context("spawn")
        # Lock hierarchy (strictly outer -> inner): _lock -> _queue_cond -> _proc_state_lock.
        self._lock = Lock()
        self._queue_cond = Condition()
        self._proc_state_lock = Lock()

        self._proc: mp.Process | None = None
        self._conn: Connection | None = None
        self._starting_proc: mp.Process | None = None
        self._timer: Timer | None = None
        self._timer_gen = 0
        self._last_used = 0.0

        self._queue: deque[tuple[tuple[Any, ...], _BatchFuture, int]] = deque()
        self._admin_queue: deque[tuple[tuple[Any, ...], _BatchFuture, int]] = deque()
        self._last_plane = "admin"
        self._in_flight = 0
        self._closed = False
        self._dispatcher: Thread | None = None

    def is_loaded(self) -> bool:
        with self._proc_state_lock:
            proc = self._proc
            if proc is None:
                return False
            try:
                return proc.is_alive()
            except (ValueError, AttributeError):
                return False

    def start(self) -> None:
        with self._lock:
            with self._queue_cond:
                if self._closed:
                    raise RuntimeError("Model worker is closed.")
                busy = self._has_pending_work_locked()
            self._cancel_timer_locked()
            self._ensure_worker_locked()
            self._last_used = time.monotonic()
            if not busy:
                self._schedule_timer_locked()

    def classify_batch(
        self,
        items: Sequence[tuple[str, list[str] | dict[str, str | None], str, str | None]],
        *,
        cancel_event: Event | None = None,
        timeout: float | None = None,
        plane: str = "interactive",
    ) -> list[tuple[str, Any]]:
        """Enqueue a batch of classification items and return per-item status tuples."""
        if plane not in _VALID_PLANES:
            raise ValueError(f"Unsupported plane {plane!r}; expected 'interactive' or 'admin'.")
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be 0 or greater.")

        normalized: list[tuple[Any, ...]] = []
        for raw_item in items:
            item = tuple(raw_item)
            if len(item) != 4:
                raise ValueError("Each classification item must be a 4-tuple of (text, labels, task, prompt).")
            normalized.append(item)

        if cancel_event is not None and cancel_event.is_set():
            raise WorkerCancelledError("Classification request was cancelled.")

        deadline = (time.monotonic() + float(timeout)) if timeout is not None else None

        with self._queue_cond:
            if self._closed:
                raise RuntimeError("Model worker is closed.")
            if not normalized:
                return []
            if cancel_event is not None and cancel_event.is_set():
                raise WorkerCancelledError("Classification request was cancelled.")
            future = _BatchFuture(len(normalized), cancel_event=cancel_event, deadline=deadline)
            self._ensure_dispatcher_locked()
            target_queue = self._admin_queue if plane == "admin" else self._queue
            for idx, item in enumerate(normalized):
                target_queue.append((item, future, idx))
            self._queue_cond.notify_all()

        self._wait_for_future(future, cancel_event=cancel_event, deadline=deadline)

        if future.fatal_error is not None:
            raise future.fatal_error
        if future.is_cancelled():
            raise WorkerCancelledError("Classification request was cancelled.")
        return [
            res if res is not None else ("error", "Model worker did not produce a result.")
            for res in future.results
        ]

    def _wait_for_future(
        self,
        future: _BatchFuture,
        *,
        cancel_event: Event | None,
        deadline: float | None,
    ) -> None:
        if cancel_event is None and deadline is None:
            future.done.wait()
            return

        while True:
            if cancel_event is not None and cancel_event.is_set():
                if future.cancel():
                    with self._queue_cond:
                        self._queue_cond.notify_all()
                return
            wait_slice = _CANCEL_WAIT_POLL_INTERVAL
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if future.cancel():
                        with self._queue_cond:
                            self._queue_cond.notify_all()
                    return
                wait_slice = min(wait_slice, remaining)
            if future.done.wait(timeout=wait_slice):
                return

    def classify(
        self,
        text: str,
        labels: list[str] | dict[str, str | None],
        task: str = "decision",
        prompt: str | None = None,
        *,
        cancel_event: Event | None = None,
        timeout: float | None = None,
        plane: str = "interactive",
    ) -> tuple[str, dict[str, float]]:
        status, payload = self.classify_batch(
            [(text, labels, task, prompt)],
            cancel_event=cancel_event,
            timeout=timeout,
            plane=plane,
        )[0]
        if status == "ok":
            if isinstance(payload, tuple) and len(payload) >= 2:
                return payload[0], payload[1]
            return payload
        if status == "input_too_long":
            raise WorkerInputTooLongError(payload)
        raise RuntimeError(payload)

    def _terminate_proc_best_effort(self, proc: Any, *, force_kill: bool = False) -> None:
        if proc is None:
            return
        with self._proc_state_lock:
            try:
                if not proc.is_alive():
                    return
                if force_kill:
                    proc.kill()
                else:
                    terminate = getattr(proc, "terminate", None)
                    if callable(terminate):
                        terminate()
                    else:
                        proc.kill()
            except (OSError, ValueError):
                pass

    def _cleanup_process(self, proc: Any, *, terminate_first: bool = True, graceful_timeout: float = 2.0) -> None:
        try:
            if terminate_first:
                self._terminate_proc_best_effort(proc, force_kill=False)
            proc.join(timeout=graceful_timeout)
            self._terminate_proc_best_effort(proc, force_kill=True)
            proc.join(timeout=2.0)
        except (OSError, ValueError):
            pass
        finally:
            with self._proc_state_lock:
                try:
                    proc.close()
                except (OSError, ValueError):
                    pass

    def close(self) -> None:
        with self._queue_cond:
            self._closed = True
            pending_futures: list[_BatchFuture] = []
            seen_ids: set[int] = set()
            for q in (self._queue, self._admin_queue):
                while q:
                    _, fut, _ = q.popleft()
                    if fut is not None and id(fut) not in seen_ids:
                        seen_ids.add(id(fut))
                        pending_futures.append(fut)
            dispatcher = self._dispatcher
            self._queue_cond.notify_all()

        for fut in pending_futures:
            fut.set_fatal_error(RuntimeError("Model worker is closed."))

        # Immediately terminate any in-progress startup child without waiting for _lock.
        # Sole ownership of closing parent_conn remains on the startup thread to avoid POSIX FD double-close races.
        with self._proc_state_lock:
            starting_proc = self._starting_proc
        self._terminate_proc_best_effort(starting_proc, force_kill=False)

        if dispatcher is not None and dispatcher.is_alive():
            dispatcher.join(timeout=5.0)
            if dispatcher.is_alive():
                logging.warning("Model worker dispatcher did not exit within 5s during shutdown; killing child process")
                with self._proc_state_lock:
                    proc = self._proc or self._starting_proc
                self._terminate_proc_best_effort(proc, force_kill=True)
                dispatcher.join(timeout=2.0)

        if not self._lock.acquire(timeout=5.0):
            logging.warning("Model worker lock busy during shutdown; killing child process")
            with self._proc_state_lock:
                stuck_proc = self._proc or self._starting_proc
            self._terminate_proc_best_effort(stuck_proc, force_kill=True)
            self._lock.acquire()
        try:
            self._cancel_timer_locked()
            self._stop_worker_locked("server shutdown")
        finally:
            self._lock.release()

    def set_idle_timeout(self, idle_timeout: float) -> None:
        """Dynamically update idle eviction timeout and reschedule or evict immediately if needed."""
        new_timeout = float(idle_timeout)
        if not math.isfinite(new_timeout) or new_timeout < 0:
            raise ValueError("IDLE_TIMEOUT_SECONDS must be 0 or greater.")
        with self._lock:
            self._cancel_timer_locked()
            self.idle_timeout = new_timeout
            if self._proc is None:
                return
            if not self._proc.is_alive():
                self._stop_worker_locked("stale worker cleanup")
                return
            if self.idle_timeout <= 0:
                return
            with self._queue_cond:
                if self._has_pending_work_locked():
                    return
            idle_for = time.monotonic() - self._last_used
            remaining = self.idle_timeout - idle_for
            if remaining <= 0.05:
                self._stop_worker_locked(f"idle for {idle_for:.0f}s")
            else:
                self._schedule_timer_locked(remaining)

    def _has_pending_work_locked(self) -> bool:
        return bool(self._queue) or bool(self._admin_queue) or self._in_flight > 0

    def _total_queued_locked(self) -> int:
        return len(self._queue) + len(self._admin_queue)

    def _ensure_dispatcher_locked(self) -> None:
        if self._dispatcher is not None and self._dispatcher.is_alive():
            return
        dispatcher = Thread(target=self._dispatcher_loop, name="choice-worker-dispatcher", daemon=True)
        self._dispatcher = dispatcher
        dispatcher.start()

    @staticmethod
    def _prune_cancelled_head_locked(q: deque[tuple[tuple[Any, ...], _BatchFuture, int]]) -> None:
        while q and (q[0][1] is None or q[0][1].is_done_or_cancelled()):
            q.popleft()

    def _pop_batch_locked(self) -> list[tuple[tuple[Any, ...], _BatchFuture, int]]:
        self._prune_cancelled_head_locked(self._queue)
        self._prune_cancelled_head_locked(self._admin_queue)

        has_interactive = bool(self._queue)
        has_admin = bool(self._admin_queue)
        if not has_interactive and not has_admin:
            return []

        if has_interactive and has_admin:
            chosen_plane = "interactive" if self._last_plane == "admin" else "admin"
        elif has_interactive:
            chosen_plane = "interactive"
        else:
            chosen_plane = "admin"

        self._last_plane = chosen_plane
        if chosen_plane == "admin":
            q = self._admin_queue
            limit = min(self.max_batch_size, self.max_admin_batch_size)
        else:
            q = self._queue
            limit = self.max_batch_size

        batch: list[tuple[tuple[Any, ...], _BatchFuture, int]] = []
        long_count = 0
        total_chars = 0
        while q and len(batch) < limit:
            entry = q[0]
            fut = entry[1]
            if fut is None or fut.is_done_or_cancelled():
                q.popleft()
                continue
            text = entry[0][0]
            text_len = len(text) if isinstance(text, str) else 0
            is_long = text_len > self.long_text_chars
            if batch:
                if is_long and (
                    long_count >= self.max_long_items_per_batch or self.max_long_items_per_batch == 1
                ):
                    break
                if total_chars + text_len > self.max_batch_chars:
                    break
            q.popleft()
            batch.append(entry)
            total_chars += text_len
            if is_long:
                long_count += 1
                if long_count >= self.max_long_items_per_batch:
                    break
        return batch

    def _reschedule_idle_if_needed(self) -> None:
        with self._lock:
            if self._proc is None or self.idle_timeout <= 0 or self._timer is not None:
                return
            with self._queue_cond:
                if self._closed or self._has_pending_work_locked():
                    return
            idle_for = time.monotonic() - self._last_used
            remaining = self.idle_timeout - idle_for
            if remaining <= 0.05:
                self._stop_worker_locked(f"idle for {idle_for:.0f}s")
            else:
                self._schedule_timer_locked(remaining)

    def _dispatcher_loop(self) -> None:
        while True:
            batch: list[tuple[tuple[Any, ...], _BatchFuture, int]] = []
            need_idle_check = False
            with self._queue_cond:
                while not (self._queue or self._admin_queue) and not self._closed:
                    self._queue_cond.wait()
                if self._closed and not (self._queue or self._admin_queue):
                    return

                if (
                    self.coalesce_ms > 0
                    and self._total_queued_locked() < self.max_batch_size
                    and not self._closed
                ):
                    deadline = time.monotonic() + (self.coalesce_ms / 1000.0)
                    while self._total_queued_locked() < self.max_batch_size and not self._closed:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._queue_cond.wait(timeout=remaining)

                if self._closed:
                    for q in (self._queue, self._admin_queue):
                        while q:
                            _, fut, _ = q.popleft()
                            if fut is not None:
                                fut.set_fatal_error(RuntimeError("Model worker is closed."))
                    return

                batch = self._pop_batch_locked()
                if not batch:
                    need_idle_check = True
                else:
                    self._in_flight += len(batch)

            if need_idle_check:
                self._reschedule_idle_if_needed()
                continue

            self._execute_batch(batch)

    def _fail_batch_futures(self, batch: list[tuple[tuple[Any, ...], _BatchFuture, int]], message: str) -> None:
        seen_ids: set[int] = set()
        for _, fut, _ in batch:
            if id(fut) not in seen_ids:
                seen_ids.add(id(fut))
                fut.set_fatal_error(RuntimeError(message))

    def _execute_batch(self, batch: list[tuple[tuple[Any, ...], _BatchFuture, int]]) -> None:
        active_batch = batch
        batch_results: list[tuple[str, Any]] | None = None
        failure_message: str | None = None

        with self._lock:
            try:
                self._cancel_timer_locked()
                with self._queue_cond:
                    closed = self._closed
                if closed:
                    failure_message = "Model worker is closed."
                else:
                    active_batch = [entry for entry in batch if not entry[1].is_done_or_cancelled()]
                    if active_batch:
                        self._ensure_worker_locked()
                        # Re-filter in case callers cancelled while cold startup was in progress.
                        active_batch = [entry for entry in active_batch if not entry[1].is_done_or_cancelled()]
                    if active_batch:
                        batch_items = [item for item, _, _ in active_batch]
                        assert self._conn is not None
                        try:
                            self._conn.send(("classify_batch", batch_items))
                            status, payload = self._conn.recv()
                        except (EOFError, OSError):
                            self._stop_worker_locked("unexpected worker exit")
                            failure_message = "Model worker exited unexpectedly."
                        else:
                            if (
                                status == "batch_ok"
                                and isinstance(payload, list)
                                and len(payload) == len(active_batch)
                            ):
                                batch_results = payload
                            elif status == "batch_error":
                                self._stop_worker_locked("worker batch error")
                                failure_message = str(payload)
                            else:
                                self._stop_worker_locked("unexpected worker IPC response")
                                failure_message = "Model worker returned an unexpected IPC response."
                        self._last_used = time.monotonic()
            except Exception as exc:
                if self._proc is not None or self._conn is not None:
                    self._stop_worker_locked("worker IPC error")
                failure_message = str(exc) or f"{type(exc).__name__}"
            finally:
                with self._queue_cond:
                    self._in_flight -= len(batch)
                    still_busy = self._has_pending_work_locked() or self._closed
                if not still_busy and self._proc is not None:
                    self._last_used = time.monotonic()
                    self._schedule_timer_locked()

        if failure_message is not None:
            self._fail_batch_futures(active_batch, failure_message)
            return

        if batch_results is None:
            return

        for (_, fut, idx), entry_result in zip(active_batch, batch_results, strict=True):
            fut.set_item_result(idx, entry_result)

    def _ensure_worker_locked(self) -> None:
        if self._proc is not None and self._proc.is_alive() and self._conn is not None:
            return
        if self._proc is not None or self._conn is not None:
            self._stop_worker_locked("stale worker cleanup")

        with self._queue_cond:
            if self._closed:
                raise RuntimeError("Model worker is closed.")

        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, self.model_path, self.device, self.precision),
            daemon=True,
        )
        try:
            proc.start()
        except BaseException:
            for c in (child_conn, parent_conn):
                try:
                    c.close()
                except (OSError, ValueError):
                    pass
            try:
                proc.close()
            except (OSError, ValueError):
                pass
            raise

        try:
            try:
                child_conn.close()
            except (OSError, ValueError):
                pass

            with self._proc_state_lock:
                self._starting_proc = proc

            deadline = time.monotonic() + self.startup_timeout
            while True:
                with self._queue_cond:
                    if self._closed:
                        raise RuntimeError("Model worker is closed.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"Model worker startup timed out after {self.startup_timeout:.1f}s."
                    )
                slice_timeout = min(_STARTUP_POLL_INTERVAL, remaining)
                try:
                    has_data = parent_conn.poll(slice_timeout)
                except (EOFError, OSError) as exc:
                    with self._queue_cond:
                        if self._closed:
                            raise RuntimeError("Model worker is closed.") from exc
                    raise RuntimeError("Model worker failed during startup.") from exc
                if has_data:
                    break

            with self._queue_cond:
                if self._closed:
                    raise RuntimeError("Model worker is closed.")

            try:
                startup_msg = parent_conn.recv()
            except (EOFError, OSError) as exc:
                with self._queue_cond:
                    if self._closed:
                        raise RuntimeError("Model worker is closed.") from exc
                raise RuntimeError("Model worker failed during startup.") from exc

            if not isinstance(startup_msg, tuple) or len(startup_msg) != 2:
                raise RuntimeError("Model worker returned an invalid startup response.")
            status, detail = startup_msg
            if status != "ready":
                raise RuntimeError(f"Model worker startup failed: {detail}")

            with self._queue_cond:
                if self._closed:
                    raise RuntimeError("Model worker is closed.")
                with self._proc_state_lock:
                    self._proc = proc
                    self._starting_proc = None
            self._conn = parent_conn
        except BaseException:
            with self._proc_state_lock:
                if self._starting_proc is proc:
                    self._starting_proc = None
            try:
                parent_conn.close()
            except (OSError, ValueError):
                pass
            self._cleanup_process(proc, terminate_first=True, graceful_timeout=2.0)
            raise

        self._last_used = time.monotonic()
        logging.info("Model worker started: pid=%s %s", proc.pid, detail)

    def _cancel_timer_locked(self) -> None:
        self._timer_gen += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule_timer_locked(self, delay: float | None = None) -> None:
        if self.idle_timeout <= 0:
            return
        wait_seconds = self.idle_timeout if delay is None else max(0.05, delay)
        self._timer_gen += 1
        gen = self._timer_gen
        timer = Timer(wait_seconds, self._unload_if_idle, args=(gen,))
        timer.daemon = True
        timer.start()
        self._timer = timer

    def _unload_if_idle(self, timer_gen: int | None = None) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            if timer_gen is not None and timer_gen != self._timer_gen:
                return
            if self._proc is None or self.idle_timeout <= 0:
                return
            with self._queue_cond:
                if self._has_pending_work_locked():
                    return
            idle_for = time.monotonic() - self._last_used
            remaining = self.idle_timeout - idle_for
            if remaining > 0.05:
                self._schedule_timer_locked(remaining)
                return
            self._timer = None
            self._stop_worker_locked(f"idle for {idle_for:.0f}s")
        finally:
            self._lock.release()

    def _stop_worker_locked(self, reason: str) -> None:
        conn = self._conn
        with self._proc_state_lock:
            proc = self._proc
            self._proc = None
        self._conn = None

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
            with self._proc_state_lock:
                proc.close()
            logging.info("Model worker stopped (%s): pid=%s GPU context released", reason, pid)
