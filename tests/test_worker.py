"""Concurrency, dynamic micro-batching, idle eviction, and crash recovery tests for ModelWorker."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import multiprocessing as mp
import pickle
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from app.worker import ModelWorker, WorkerCancelledError, WorkerInputTooLongError, _worker_main


class _PipeChannel:
    """One unidirectional in-memory channel supporting poll(), recv(), send(), and EOF."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.items: deque[object] = deque()
        self.eof = False


class _PipeEndpoint:
    """In-memory bidirectional pipe endpoint with POSIX-like dup/close/poll/EOF semantics."""

    def __init__(
        self,
        send_ch: _PipeChannel,
        recv_ch: _PipeChannel,
        send_refcount: list[int],
        lock: threading.Lock,
    ) -> None:
        self._send_ch = send_ch
        self._recv_ch = recv_ch
        self._send_refcount = send_refcount
        self._lock = lock
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def dup(self) -> _PipeEndpoint:
        with self._lock:
            if self._closed:
                raise OSError("Cannot dup closed pipe endpoint")
            self._send_refcount[0] += 1
            return _PipeEndpoint(self._send_ch, self._recv_ch, self._send_refcount, self._lock)

    def send(self, obj: object) -> None:
        with self._lock:
            if self._closed:
                raise OSError("Pipe endpoint is closed")
        with self._send_ch.cond:
            self._send_ch.items.append(obj)
            self._send_ch.cond.notify_all()

    def poll(self, timeout: float = 0.0) -> bool:
        if self._closed:
            raise OSError("Pipe endpoint is closed")
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._recv_ch.cond:
            while not self._recv_ch.items and not self._recv_ch.eof and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._recv_ch.cond.wait(timeout=remaining)
            if self._closed:
                raise OSError("Pipe endpoint is closed")
            return bool(self._recv_ch.items or self._recv_ch.eof)

    def recv(self) -> object:
        if self._closed:
            raise OSError("Pipe endpoint is closed")
        deadline = time.monotonic() + 5.0
        with self._recv_ch.cond:
            while not self._recv_ch.items and not self._recv_ch.eof and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Fake pipe recv timed out")
                self._recv_ch.cond.wait(timeout=remaining)
            if self._closed:
                raise OSError("Pipe endpoint is closed")
            if self._recv_ch.items:
                return self._recv_ch.items.popleft()
            raise EOFError("Pipe closed by peer")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._send_refcount[0] -= 1
            mark_eof = self._send_refcount[0] == 0
        if mark_eof:
            with self._send_ch.cond:
                self._send_ch.eof = True
                self._send_ch.cond.notify_all()
        with self._recv_ch.cond:
            self._recv_ch.cond.notify_all()


class _FakeWorkerIPC:
    """Thread-based fake spawn context simulating the child CUDA process over in-memory pipes."""

    def __init__(
        self,
        *,
        infer_delay: float = 0.0,
        crash_on_call: int | None = None,
        startup_fail: str | None = None,
        startup_gate: threading.Event | None = None,
        startup_recv_exc: BaseException | None = None,
        ignore_terminate: bool = False,
    ) -> None:
        self.infer_delay = infer_delay
        self.crash_on_call = crash_on_call
        self.startup_fail = startup_fail
        self.startup_gate = startup_gate
        self.startup_recv_exc = startup_recv_exc
        self.ignore_terminate = ignore_terminate
        self.spawn_count = 0
        self.batch_sizes: list[int] = []
        self.ipc_messages: list[object] = []
        self.terminated_count = 0
        self.killed_count = 0
        self.closed_procs = 0
        self.parent_conns: list[_PipeEndpoint] = []
        self.procs: list[object] = []
        self.startup_entered = threading.Event()
        self.active_infer_started = threading.Event()
        self.allow_infer_finish: threading.Event | None = None
        self._lock = threading.Lock()

    def Pipe(self) -> tuple[_PipeEndpoint, _PipeEndpoint]:
        p_to_c = _PipeChannel()
        c_to_p = _PipeChannel()
        lock = threading.Lock()
        parent = _PipeEndpoint(p_to_c, c_to_p, [1], lock)
        child = _PipeEndpoint(c_to_p, p_to_c, [1], lock)
        if self.startup_recv_exc is not None:
            exc = self.startup_recv_exc
            self.startup_recv_exc = None
            orig_recv = parent.recv

            def _failing_recv():
                raise exc

            parent.recv = _failing_recv  # type: ignore[method-assign]
            del orig_recv
        with self._lock:
            self.parent_conns.append(parent)
        return parent, child

    def Process(self, target, args, daemon=True):
        raw_child_conn, model_path, device, precision = args
        child_conn = raw_child_conn.dup()
        del target, model_path, device, precision, daemon
        ipc = self

        class _Proc:
            def __init__(self) -> None:
                with ipc._lock:
                    ipc.spawn_count += 1
                    self.pid = 40000 + ipc.spawn_count
                self._alive = False
                self._stop_ev = threading.Event()
                self.closed = False
                self._thread = threading.Thread(target=self._run, daemon=True)
                with ipc._lock:
                    ipc.procs.append(self)

            def _run(self) -> None:
                ipc.startup_entered.set()
                if ipc.startup_gate is not None:
                    while not ipc.startup_gate.is_set() and not self._stop_ev.is_set():
                        ipc.startup_gate.wait(timeout=0.02)
                    if self._stop_ev.is_set():
                        try:
                            child_conn.close()
                        except OSError:
                            pass
                        self._alive = False
                        return

                if ipc.startup_fail is not None:
                    try:
                        child_conn.send(("startup_error", ipc.startup_fail))
                    except OSError:
                        pass
                    finally:
                        child_conn.close()
                    self._alive = False
                    return

                try:
                    child_conn.send(("ready", "device=cpu dtype=torch.float32 torch=fake"))
                except OSError:
                    child_conn.close()
                    self._alive = False
                    return

                while not self._stop_ev.is_set():
                    try:
                        msg = child_conn.recv()
                    except (EOFError, OSError, TimeoutError):
                        break
                    if msg is None:
                        break
                    with ipc._lock:
                        ipc.ipc_messages.append(msg)
                        call_idx = len(ipc.ipc_messages)
                    if ipc.crash_on_call is not None and call_idx == ipc.crash_on_call:
                        child_conn.close()
                        self._alive = False
                        return

                    _, batch_items = msg
                    with ipc._lock:
                        ipc.batch_sizes.append(len(batch_items))

                    ipc.active_infer_started.set()
                    if ipc.allow_infer_finish is not None:
                        ipc.allow_infer_finish.wait(timeout=5.0)
                    if ipc.infer_delay > 0:
                        time.sleep(ipc.infer_delay)

                    out = []
                    for text, labels, _task, _prompt in batch_items:
                        if "__TOO_LONG__" in text:
                            out.append(("input_too_long", "Prepared input length (8100 tokens) exceeds the 7,999-token limit."))
                        elif "__ERR__" in text:
                            out.append(("error", "Model returned invalid classification scores."))
                        else:
                            names = list(labels)
                            p = 1.0 / len(names)
                            out.append(("ok", (names[0], {n: p for n in names})))
                    try:
                        child_conn.send(("batch_ok", out))
                    except (EOFError, OSError):
                        break

                try:
                    child_conn.close()
                except OSError:
                    pass
                self._alive = False

            def start(self) -> None:
                self._alive = True
                self._thread.start()

            def is_alive(self) -> bool:
                return self._alive and self._thread.is_alive()

            def join(self, timeout=None) -> None:
                self._thread.join(timeout=timeout)
                if not self._thread.is_alive():
                    self._alive = False

            def terminate(self) -> None:
                with ipc._lock:
                    ipc.terminated_count += 1
                if ipc.ignore_terminate:
                    return
                self._stop_ev.set()
                if ipc.allow_infer_finish is not None:
                    ipc.allow_infer_finish.set()
                try:
                    child_conn.close()
                except OSError:
                    pass

            def kill(self) -> None:
                with ipc._lock:
                    ipc.killed_count += 1
                self._stop_ev.set()
                if ipc.allow_infer_finish is not None:
                    ipc.allow_infer_finish.set()
                try:
                    child_conn.close()
                except OSError:
                    pass
                self._alive = False

            def close(self) -> None:
                self.closed = True
                with ipc._lock:
                    ipc.closed_procs += 1

        return _Proc()


class ModelWorkerMicroBatchTests(unittest.TestCase):
    def test_concurrent_classify_coalescing_and_bulk_50_items(self) -> None:
        ipc = _FakeWorkerIPC()
        with tempfile.TemporaryDirectory() as tmpdir:
            w = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0, coalesce_ms=30.0, max_batch_size=32)
            w._ctx = ipc  # type: ignore[assignment]
            try:
                # 1. Coalesce 8 concurrent classify() calls into 1 IPC micro-batch
                barrier = threading.Barrier(8)
                results: dict[int, tuple[str, dict[str, float]]] = {}
                errors: list[BaseException] = []

                def _call(idx: int) -> None:
                    try:
                        barrier.wait(timeout=3.0)
                        results[idx] = w.classify(f"item-{idx}", ["a", "b"])
                    except BaseException as exc:
                        errors.append(exc)

                threads = [threading.Thread(target=_call, args=(i,)) for i in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=5.0)

                self.assertEqual(errors, [])
                self.assertEqual(len(results), 8)
                self.assertEqual(ipc.batch_sizes, [8])
                self.assertEqual(ipc.spawn_count, 1)

                # 2. Bulk classify_batch of 50 items splits into 32 + 18 (2 IPC calls) with partial error isolation
                ipc.batch_sizes.clear()
                bulk = []
                for i in range(50):
                    if i == 7:
                        text = "__TOO_LONG__ entry"
                    elif i == 41:
                        text = "__ERR__ entry"
                    else:
                        text = f"log line {i}"
                    bulk.append((text, ["normal", "alert"], "triage", "Check log"))

                bulk_res = w.classify_batch(bulk)
                self.assertEqual(len(bulk_res), 50)
                self.assertEqual(ipc.batch_sizes, [32, 18])
                self.assertEqual(bulk_res[0][0], "ok")
                self.assertEqual(bulk_res[7][0], "input_too_long")
                self.assertEqual(bulk_res[41][0], "error")
                self.assertEqual(bulk_res[49][0], "ok")

                # 3. Single classify() error mapping
                with self.assertRaises(WorkerInputTooLongError):
                    w.classify("__TOO_LONG__", ["a", "b"])
                with self.assertRaises(RuntimeError):
                    w.classify("__ERR__", ["a", "b"])
            finally:
                w.close()

    def test_idle_timer_never_evicts_while_in_flight_and_evicts_after_idle(self) -> None:
        ipc = _FakeWorkerIPC()
        ipc.allow_infer_finish = threading.Event()
        with tempfile.TemporaryDirectory() as tmpdir:
            w = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=0.15, coalesce_ms=0.0, max_batch_size=32)
            w._ctx = ipc  # type: ignore[assignment]
            try:
                holder: dict[str, object] = {}

                def _run() -> None:
                    holder["res"] = w.classify("slow item", ["a", "b"])

                t = threading.Thread(target=_run)
                t.start()
                self.assertTrue(ipc.active_infer_started.wait(timeout=3.0))

                # Force _last_used far in the past and invoke _unload_if_idle while request is in-flight!
                w._last_used = time.monotonic() - 100.0
                w._unload_if_idle()
                self.assertTrue(w.is_loaded(), "Idle timer must NOT evict while request is in-flight")

                # Release in-flight request; worker remains loaded immediately after and schedules timer
                ipc.allow_infer_finish.set()
                t.join(timeout=3.0)
                self.assertIn("res", holder)
                self.assertTrue(w.is_loaded())
                self.assertIsNotNone(w._timer)

                # Wait past idle_timeout=0.15s -> worker auto-evicts!
                deadline = time.monotonic() + 2.0
                while w.is_loaded() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(w.is_loaded())

                # Also verify queued (not yet holding _lock) work blocks _unload_if_idle eviction
                w._proc = types.SimpleNamespace(is_alive=lambda: True, pid=123, join=lambda timeout=None: None, kill=lambda: None, close=lambda: None)  # type: ignore[assignment]
                w._conn = types.SimpleNamespace(send=lambda v: None, close=lambda: None)  # type: ignore[assignment]
                w._last_used = time.monotonic() - 100.0
                with w._queue_cond:
                    w._queue.append((("q", ["a", "b"], "decision", None), None, 0))  # type: ignore[arg-type]
                w._unload_if_idle()
                self.assertTrue(w.is_loaded(), "Idle timer must NOT evict while item is queued")
                with w._queue_cond:
                    w._queue.clear()
                w._unload_if_idle()
                self.assertFalse(w.is_loaded())
            finally:
                ipc.allow_infer_finish.set()
                w.close()

    def test_crash_recovery_and_startup_failure(self) -> None:
        ipc = _FakeWorkerIPC(crash_on_call=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            w = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0, coalesce_ms=15.0, max_batch_size=32)
            w._ctx = ipc  # type: ignore[assignment]
            try:
                # Two concurrent callers share the crashed batch -> both fail cleanly with RuntimeError
                barrier = threading.Barrier(2)
                errs: list[BaseException] = []

                def _call(idx: int) -> None:
                    try:
                        barrier.wait(timeout=3.0)
                        w.classify(f"req-{idx}", ["a", "b"])
                    except BaseException as exc:
                        errs.append(exc)

                threads = [threading.Thread(target=_call, args=(i,)) for i in range(2)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=5.0)

                self.assertEqual(len(errs), 2)
                self.assertTrue(all(isinstance(e, RuntimeError) for e in errs))
                self.assertFalse(w.is_loaded())

                # Next request respawns cleanly and succeeds!
                choice, _ = w.classify("recovered", ["a", "b"])
                self.assertEqual(choice, "a")
                self.assertEqual(ipc.spawn_count, 2)
            finally:
                w.close()

        # Startup error test
        fail_ipc = _FakeWorkerIPC(startup_fail="RuntimeError: bad checkpoint")
        with tempfile.TemporaryDirectory() as tmpdir:
            w2 = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0, coalesce_ms=0.0)
            w2._ctx = fail_ipc  # type: ignore[assignment]
            try:
                with self.assertRaises(RuntimeError) as cm:
                    w2.classify("hello", ["a", "b"])
                self.assertIn("bad checkpoint", str(cm.exception))
                self.assertFalse(w2.is_loaded())
            finally:
                w2.close()

    def test_close_wakes_pending_callers_and_rejects_after_close(self) -> None:
        ipc = _FakeWorkerIPC()
        ipc.allow_infer_finish = threading.Event()
        with tempfile.TemporaryDirectory() as tmpdir:
            w = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0, coalesce_ms=0.0, max_batch_size=1)
            w._ctx = ipc  # type: ignore[assignment]

            # Item 1 occupies the worker; Item 2 stays queued
            err2: list[BaseException] = []
            res1: list[object] = []

            t1 = threading.Thread(target=lambda: res1.append(w.classify("first", ["a", "b"])))
            t1.start()
            self.assertTrue(ipc.active_infer_started.wait(timeout=3.0))

            t2 = threading.Thread(target=lambda: self._capture_err(err2, lambda: w.classify("queued", ["a", "b"])))
            t2.start()
            # Ensure Item 2 is queued
            deadline = time.monotonic() + 1.0
            while len(w._queue) == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(len(w._queue), 1)

            # Release Item 1 shortly after close() begins draining queue
            release = threading.Timer(0.05, ipc.allow_infer_finish.set)
            release.start()
            w.close()
            t1.join(timeout=3.0)
            t2.join(timeout=3.0)

            self.assertEqual(len(res1), 1)
            self.assertEqual(len(err2), 1)
            self.assertIsInstance(err2[0], RuntimeError)
            self.assertIn("closed", str(err2[0]).lower())
            self.assertFalse(w.is_loaded())

            with self.assertRaises(RuntimeError):
                w.classify("after-close", ["a", "b"])
            with self.assertRaises(RuntimeError):
                w.classify_batch([])
            with self.assertRaises(RuntimeError):
                w.start()

    @staticmethod
    def _capture_err(bucket: list[BaseException], fn) -> None:
        try:
            fn()
        except BaseException as exc:
            bucket.append(exc)

    def test_worker_main_pipe_protocol(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "2.7.1+cu126-fake"

        @contextmanager
        def _inf():
            yield

        fake_torch.inference_mode = _inf

        fake_model_mod = types.ModuleType("app.model")
        fake_model_mod.resolve_dtype = lambda device, precision: "float32"
        fake_model_mod.load_model = lambda model_path, device, precision: {"path": model_path}

        def _fake_classify_batch(bundle, items):
            assert bundle == {"path": "/tmp/amber"}
            return [("ok", ("a", {"a": 0.8, "b": 0.2}, 42)) for _ in items]

        fake_model_mod.classify_batch = _fake_classify_batch

        parent_conn, child_conn = mp.Pipe()
        with patch.dict(sys.modules, {"torch": fake_torch, "app.model": fake_model_mod}):
            t = threading.Thread(
                target=_worker_main,
                args=(child_conn, "/tmp/amber", "cpu", "fp32"),
                daemon=True,
            )
            t.start()
            self.assertTrue(parent_conn.poll(3.0))
            status, detail = parent_conn.recv()
            self.assertEqual(status, "ready")
            self.assertIn("device=cpu", detail)

            parent_conn.send(("classify_batch", [("hello", ["a", "b"], "decision", None)]))
            self.assertTrue(parent_conn.poll(3.0))
            r_status, r_payload = parent_conn.recv()
            self.assertEqual(r_status, "batch_ok")
            self.assertEqual(r_payload, [("ok", ("a", {"a": 0.8, "b": 0.2}, 42))])

            parent_conn.send(None)
            parent_conn.close()
            t.join(timeout=3.0)
            self.assertFalse(t.is_alive())

    def test_ipc_non_oserror_cleanup_and_multichunk_crash_and_forced_close(self) -> None:
        # 1. Non-OSError IPC exception (e.g. ValueError/UnpicklingError) stops worker & respawns cleanly
        ipc = _FakeWorkerIPC()
        with tempfile.TemporaryDirectory() as tmpdir:
            w = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0, coalesce_ms=0.0)
            w._ctx = ipc  # type: ignore[assignment]
            try:
                self.assertEqual(w.classify("ok1", ["a", "b"])[0], "a")
                self.assertIsNotNone(w._conn)
                orig_recv = w._conn.recv
                w._conn.recv = lambda: (_ for _ in ()).throw(ValueError("corrupt pickle frame"))  # type: ignore[method-assign]
                with self.assertRaises(RuntimeError) as cm:
                    w.classify("corrupt", ["a", "b"])
                self.assertIn("corrupt pickle frame", str(cm.exception))
                self.assertFalse(w.is_loaded())
                self.assertIsNone(w._conn)
                del orig_recv

                # Respawns cleanly on next call
                self.assertEqual(w.classify("ok2", ["a", "b"])[0], "a")
                self.assertEqual(ipc.spawn_count, 2)
            finally:
                w.close()

        # 2. Multi-chunk 50-item batch fails cleanly if chunk 1 crashes without hanging on chunk 2
        crash_ipc = _FakeWorkerIPC(crash_on_call=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            w2 = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0, coalesce_ms=0.0, max_batch_size=32)
            w2._ctx = crash_ipc  # type: ignore[assignment]
            try:
                items = [(f"log-{i}", ["a", "b"], "decision", None) for i in range(50)]
                with self.assertRaises(RuntimeError):
                    w2.classify_batch(items)
                self.assertFalse(w2.is_loaded())
            finally:
                w2.close()

        # 3. close() timeout fallback kills wedged child process to unblock dispatcher
        wedged_ipc = _FakeWorkerIPC()
        wedged_ipc.allow_infer_finish = threading.Event()
        with tempfile.TemporaryDirectory() as tmpdir:
            w3 = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0, coalesce_ms=0.0)
            w3._ctx = wedged_ipc  # type: ignore[assignment]
            err: list[BaseException] = []
            t = threading.Thread(target=lambda: self._capture_err(err, lambda: w3.classify("stuck", ["a", "b"])))
            t.start()
            self.assertTrue(wedged_ipc.active_infer_started.wait(timeout=3.0))

            orig_join = threading.Thread.join

            def _fast_timeout_join(self_thread, timeout=None):
                if self_thread is w3._dispatcher and timeout == 5.0:
                    return
                return orig_join(self_thread, timeout=timeout)

            with patch.object(threading.Thread, "join", _fast_timeout_join):
                w3.close()
            t.join(timeout=3.0)
            self.assertGreaterEqual(wedged_ipc.killed_count, 1)
            self.assertEqual(len(err), 1)
            self.assertFalse(w3.is_loaded())

    def test_startup_close_race_and_startup_timeout_cleanup(self) -> None:
        # 1. close() while start() is blocked in child startup unblocks immediately without deadlocking on _lock
        gate = threading.Event()
        ipc = _FakeWorkerIPC(startup_gate=gate)
        with tempfile.TemporaryDirectory() as tmpdir:
            w = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0, startup_timeout=30.0)
            w._ctx = ipc  # type: ignore[assignment]
            start_errs: list[BaseException] = []
            t = threading.Thread(target=lambda: self._capture_err(start_errs, w.start))
            t.start()
            self.assertTrue(ipc.startup_entered.wait(timeout=3.0))

            # Wait briefly until _starting_proc is recorded; is_loaded() must stay False until "ready"
            deadline = time.monotonic() + 2.0
            while w._starting_proc is None and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIsNotNone(w._starting_proc)
            self.assertFalse(w.is_loaded())

            t0 = time.monotonic()
            w.close()
            elapsed = time.monotonic() - t0
            t.join(timeout=3.0)

            self.assertFalse(t.is_alive())
            self.assertLess(elapsed, 2.0, "close() must not block on startup_timeout or deadlock on _lock")
            self.assertEqual(len(start_errs), 1)
            self.assertIsInstance(start_errs[0], RuntimeError)
            self.assertIsNone(w._starting_proc)
            self.assertFalse(w.is_loaded())
            self.assertTrue(ipc.parent_conns[0].closed)
            self.assertTrue(ipc.procs[0].closed)  # type: ignore[attr-defined]
            self.assertGreaterEqual(ipc.terminated_count + ipc.killed_count, 1)

        # 2. startup_timeout bounds startup wait and cleans up child + pipe
        gate2 = threading.Event()
        ipc2 = _FakeWorkerIPC(startup_gate=gate2)
        with tempfile.TemporaryDirectory() as tmpdir:
            w2 = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0, startup_timeout=0.15)
            w2._ctx = ipc2  # type: ignore[assignment]
            try:
                with self.assertRaises(RuntimeError) as cm:
                    w2.start()
                self.assertIn("timed out", str(cm.exception).lower())
                self.assertIsNone(w2._starting_proc)
                self.assertFalse(w2.is_loaded())
                self.assertTrue(ipc2.parent_conns[0].closed)
                self.assertTrue(ipc2.procs[0].closed)  # type: ignore[attr-defined]
            finally:
                gate2.set()
                w2.close()

    def test_startup_non_oserror_cleans_up_proc_and_pipe(self) -> None:
        ipc = _FakeWorkerIPC(startup_recv_exc=pickle.UnpicklingError("corrupt startup pickle"))
        with tempfile.TemporaryDirectory() as tmpdir:
            w = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0, coalesce_ms=0.0)
            w._ctx = ipc  # type: ignore[assignment]
            try:
                with self.assertRaises(pickle.UnpicklingError):
                    w.start()
                self.assertFalse(w.is_loaded())
                self.assertIsNone(w._starting_proc)
                self.assertIsNone(w._proc)
                self.assertIsNone(w._conn)
                self.assertTrue(ipc.parent_conns[0].closed)
                self.assertTrue(ipc.procs[0].closed)  # type: ignore[attr-defined]
                self.assertFalse(ipc.procs[0].is_alive())  # type: ignore[attr-defined]

                # Subsequent call respawns cleanly without leaked/wedged state
                choice, _ = w.classify("recovered after unpickling error", ["a", "b"])
                self.assertEqual(choice, "a")
                self.assertEqual(ipc.spawn_count, 2)
            finally:
                w.close()

    def test_cancellation_and_timeout_drop_queued_items_before_gpu(self) -> None:
        ipc = _FakeWorkerIPC()
        ipc.allow_infer_finish = threading.Event()
        with tempfile.TemporaryDirectory() as tmpdir:
            w = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0, coalesce_ms=0.0, max_batch_size=1)
            w._ctx = ipc  # type: ignore[assignment]
            try:
                res1: list[object] = []
                t1 = threading.Thread(target=lambda: res1.append(w.classify("in-flight", ["a", "b"])))
                t1.start()
                self.assertTrue(ipc.active_infer_started.wait(timeout=3.0))

                # Enqueue cancelled interactive request and timed-out admin batch while item 1 is in-flight
                cancel_ev = threading.Event()
                err_cancel: list[BaseException] = []
                err_timeout: list[BaseException] = []

                t_cancel = threading.Thread(
                    target=lambda: self._capture_err(
                        err_cancel,
                        lambda: w.classify("dead-queued", ["a", "b"], cancel_event=cancel_ev),
                    )
                )
                t_timeout = threading.Thread(
                    target=lambda: self._capture_err(
                        err_timeout,
                        lambda: w.classify_batch(
                            [("dead-admin-1", ["a", "b"], "decision", None), ("dead-admin-2", ["a", "b"], "decision", None)],
                            timeout=0.05,
                            plane="admin",
                        ),
                    )
                )
                t_cancel.start()
                t_timeout.start()

                # Wait until both queued entries are registered
                deadline = time.monotonic() + 2.0
                while (len(w._queue) < 1 or len(w._admin_queue) < 2) and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertEqual(len(w._queue), 1)
                self.assertEqual(len(w._admin_queue), 2)

                # Cancel request 2 while item 1 is still blocked on GPU
                cancel_ev.set()
                t_cancel.join(timeout=2.0)
                t_timeout.join(timeout=2.0)

                self.assertFalse(t_cancel.is_alive())
                self.assertFalse(t_timeout.is_alive())
                self.assertEqual(len(err_cancel), 1)
                self.assertIsInstance(err_cancel[0], WorkerCancelledError)
                self.assertEqual(len(err_timeout), 1)
                self.assertIsInstance(err_timeout[0], WorkerCancelledError)

                # Release item 1 and ensure one final live item succeeds while cancelled items NEVER ran on GPU
                ipc.allow_infer_finish.set()
                t1.join(timeout=3.0)
                self.assertEqual(len(res1), 1)

                final_choice, _ = w.classify("live-after", ["a", "b"])
                self.assertEqual(final_choice, "a")

                seen_texts = [
                    item[0]
                    for _, batch_items in ipc.ipc_messages  # type: ignore[misc]
                    for item in batch_items
                ]
                self.assertEqual(seen_texts, ["in-flight", "live-after"])
            finally:
                ipc.allow_infer_finish.set()
                w.close()

    def test_cancelled_queued_items_still_reschedule_idle_timer_and_skip_cold_start_ipc(self) -> None:
        # 1. If Request 2 is queued when Request 1 finishes (skipping timer) and Request 2 is then cancelled
        #    before _pop_batch_locked(), the idle eviction timer MUST still be rescheduled!
        ipc = _FakeWorkerIPC()
        ipc.allow_infer_finish = threading.Event()
        with tempfile.TemporaryDirectory() as tmpdir:
            w = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=0.15, coalesce_ms=0.0, max_batch_size=1)
            w._ctx = ipc  # type: ignore[assignment]
            try:
                res1: list[object] = []
                t1 = threading.Thread(target=lambda: res1.append(w.classify("first", ["a", "b"])))
                t1.start()
                self.assertTrue(ipc.active_infer_started.wait(timeout=3.0))

                cancel_ev = threading.Event()
                err2: list[BaseException] = []
                t2 = threading.Thread(
                    target=lambda: self._capture_err(
                        err2,
                        lambda: w.classify("cancelled-second", ["a", "b"], cancel_event=cancel_ev),
                    )
                )
                t2.start()
                deadline = time.monotonic() + 2.0
                while len(w._queue) < 1 and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertEqual(len(w._queue), 1)

                # Cancel Request 2 and release Request 1; Request 1 sees queue non-empty at finish or Request 2 is pruned
                cancel_ev.set()
                ipc.allow_infer_finish.set()
                t1.join(timeout=3.0)
                t2.join(timeout=3.0)
                self.assertEqual(len(res1), 1)
                self.assertEqual(len(err2), 1)
                self.assertIsInstance(err2[0], WorkerCancelledError)

                # Worker MUST auto-evict past idle_timeout=0.15s (idle timer must NOT be lost)
                deadline = time.monotonic() + 2.0
                while w.is_loaded() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(w.is_loaded(), "Idle eviction timer was lost after queued cancellation")
            finally:
                ipc.allow_infer_finish.set()
                w.close()

        # 2. Request cancelled during cold worker startup skips GPU IPC after startup finishes
        gate = threading.Event()
        ipc_cold = _FakeWorkerIPC(startup_gate=gate)
        with tempfile.TemporaryDirectory() as tmpdir:
            w2 = ModelWorker(tmpdir, device="cpu", precision="fp32", idle_timeout=60.0, coalesce_ms=0.0)
            w2._ctx = ipc_cold  # type: ignore[assignment]
            try:
                cancel_ev2 = threading.Event()
                err_cold: list[BaseException] = []
                t_cold = threading.Thread(
                    target=lambda: self._capture_err(
                        err_cold,
                        lambda: w2.classify("cancel-during-startup", ["a", "b"], cancel_event=cancel_ev2),
                    )
                )
                t_cold.start()
                self.assertTrue(ipc_cold.startup_entered.wait(timeout=3.0))

                cancel_ev2.set()
                t_cold.join(timeout=2.0)
                self.assertEqual(len(err_cold), 1)
                self.assertIsInstance(err_cold[0], WorkerCancelledError)

                gate.set()
                deadline = time.monotonic() + 2.0
                while not w2.is_loaded() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(ipc_cold.ipc_messages, [], "Cancelled request should not run GPU IPC after cold startup")
            finally:
                gate.set()
                w2.close()

    def test_fair_interleaving_and_long_text_ipc_splitting(self) -> None:
        # 1. Long-text requests (> long_text_chars) split to 1 per IPC batch even when coalesced concurrently
        ipc_long = _FakeWorkerIPC()
        with tempfile.TemporaryDirectory() as tmpdir:
            w = ModelWorker(
                tmpdir,
                device="cpu",
                precision="fp32",
                idle_timeout=60.0,
                coalesce_ms=30.0,
                max_batch_size=32,
                long_text_chars=4000,
                max_long_items_per_batch=1,
            )
            w._ctx = ipc_long  # type: ignore[assignment]
            try:
                barrier = threading.Barrier(4)
                errs: list[BaseException] = []

                def _run_long(idx: int) -> None:
                    try:
                        barrier.wait(timeout=3.0)
                        w.classify(f"L{idx}-" + ("x" * 4500), ["a", "b"])
                    except BaseException as exc:
                        errs.append(exc)

                threads = [threading.Thread(target=_run_long, args=(i,)) for i in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=5.0)

                self.assertEqual(errs, [])
                self.assertEqual(ipc_long.batch_sizes, [1, 1, 1, 1])
            finally:
                w.close()

        # 2. Fair interleaving prevents a bulk :9090 admin batch from starving :8000 interactive requests
        ipc_fair = _FakeWorkerIPC()
        ipc_fair.allow_infer_finish = threading.Event()
        with tempfile.TemporaryDirectory() as tmpdir:
            w2 = ModelWorker(
                tmpdir,
                device="cpu",
                precision="fp32",
                idle_timeout=60.0,
                coalesce_ms=0.0,
                max_batch_size=32,
                max_admin_batch_size=8,
            )
            w2._ctx = ipc_fair  # type: ignore[assignment]
            try:
                # Hold a blocker item in-flight so admin + interactive queues both fill before next turn
                blocker_res: list[object] = []
                t_block = threading.Thread(target=lambda: blocker_res.append(w2.classify("blocker", ["a", "b"])))
                t_block.start()
                self.assertTrue(ipc_fair.active_infer_started.wait(timeout=3.0))

                admin_res: list[object] = []
                inter_res: list[object] = []
                admin_items = [(f"admin-{i}", ["a", "b"], "decision", None) for i in range(24)]
                t_admin = threading.Thread(
                    target=lambda: admin_res.append(w2.classify_batch(admin_items, plane="admin"))
                )
                t_admin.start()

                deadline = time.monotonic() + 2.0
                while len(w2._admin_queue) < 24 and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertEqual(len(w2._admin_queue), 24)

                # Enqueue interactive request AFTER the 24-item admin batch is already queued
                t_inter = threading.Thread(
                    target=lambda: inter_res.append(w2.classify("interactive-urgent", ["a", "b"], plane="interactive"))
                )
                t_inter.start()
                deadline = time.monotonic() + 2.0
                while len(w2._queue) < 1 and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertEqual(len(w2._queue), 1)

                ipc_fair.allow_infer_finish.set()
                t_block.join(timeout=3.0)
                t_inter.join(timeout=3.0)
                t_admin.join(timeout=3.0)

                self.assertEqual(len(inter_res), 1)
                self.assertEqual(len(admin_res), 1)
                self.assertEqual(len(admin_res[0]), 24)  # type: ignore[arg-type]

                # Verify IPC batches after blocker: admin chunk 1 (8) -> interactive (1) -> admin chunk 2 (8) -> admin chunk 3 (8)
                # (blocker was interactive, so next turn is admin chunk 1, then interactive immediately interleaves!)
                batch_planes = [
                    "admin" if batch_items[0][0].startswith("admin-") else batch_items[0][0]
                    for _, batch_items in ipc_fair.ipc_messages  # type: ignore[misc]
                ]
                self.assertEqual(batch_planes[0], "blocker")
                inter_idx = batch_planes.index("interactive-urgent")
                self.assertLess(
                    inter_idx,
                    len(batch_planes) - 1,
                    f"Interactive request was starved behind all admin chunks: {batch_planes}",
                )
            finally:
                ipc_fair.allow_infer_finish.set()
                w2.close()


if __name__ == "__main__":
    unittest.main()
