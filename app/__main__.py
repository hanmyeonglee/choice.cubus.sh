"""Run the public Choice API server and optional internal Admin API server."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
import os
import signal
import sys
from typing import Iterator

import uvicorn

from .admin import AdminConfig, create_admin_app, load_admin_config, resolve_bind_addr
from .api import create_app, load_api_keys
from .runtime import MAIN_PORT, ChoiceRuntime
from .worker import validate_runtime_config


class _ManagedServer(uvicorn.Server):
    """Uvicorn server that delegates process signal handling to the shared runner."""

    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def build_servers(
    *,
    runtime: ChoiceRuntime,
    admin_config: AdminConfig,
    main_host: str = "0.0.0.0",
    main_port: int = MAIN_PORT,
    admin_host: str | None = None,
) -> list[_ManagedServer]:
    """Construct the public server and, when enabled, the internal Admin API server."""
    main_app = create_app(runtime=runtime)
    main_cfg = uvicorn.Config(
        main_app,
        host=main_host,
        port=main_port,
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
    servers: list[_ManagedServer] = [_ManagedServer(main_cfg)]

    if admin_config.enabled:
        resolved_host, resolved_port = (
            (admin_host, admin_config.port)
            if admin_host is not None
            else resolve_bind_addr(admin_config.bind_cidr, admin_config.port)
        )
        admin_app = create_admin_app(runtime=runtime, config=admin_config)
        admin_cfg = uvicorn.Config(
            admin_app,
            host=resolved_host,
            port=resolved_port,
            # Never honor X-Forwarded-For on the internal Admin API even if FORWARDED_ALLOW_IPS=* is set.
            proxy_headers=False,
            forwarded_allow_ips=None,
            server_header=False,
            ws="none",
            http="h11",
            h11_max_incomplete_event_size=16 << 10,
            limit_concurrency=32,
            timeout_keep_alive=60,
            timeout_graceful_shutdown=10,
        )
        servers.append(_ManagedServer(admin_cfg))

    return servers


async def serve_servers(servers: list[_ManagedServer]) -> None:
    """Run one or more Uvicorn servers concurrently with coordinated signal shutdown."""
    loop = asyncio.get_running_loop()
    signal_seen = False

    def handle_signal(sig: int) -> None:
        nonlocal signal_seen
        for srv in servers:
            if signal_seen and sig == signal.SIGINT:
                srv.force_exit = True
            else:
                srv.should_exit = True
        signal_seen = True

    installed_loop_signals: list[int] = []
    previous_handlers: dict[int, signal.Handlers | object] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal, sig)
            installed_loop_signals.append(sig)
        except NotImplementedError:
            previous_handlers[sig] = signal.signal(sig, lambda s, _f: handle_signal(s))

    tasks: list[asyncio.Task[None]] = []
    try:
        tasks = [asyncio.create_task(srv.serve()) for srv in servers]
        _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for srv in servers:
            srv.should_exit = True
        if pending:
            await asyncio.wait(pending)

        for task in tasks:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                raise exc
    except BaseException:
        for srv in servers:
            srv.should_exit = True
        still_running = [task for task in tasks if not task.done()]
        if still_running:
            _, timed_out = await asyncio.wait(still_running, timeout=5.0)
            for task in timed_out:
                task.cancel()
            if timed_out:
                await asyncio.gather(*timed_out, return_exceptions=True)
        raise
    finally:
        for sig in installed_loop_signals:
            loop.remove_signal_handler(sig)
        for sig, prev in previous_handlers.items():
            signal.signal(sig, prev)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        load_api_keys()
        runtime = ChoiceRuntime.from_env()
        validate_runtime_config(
            os.environ.get("MODEL_PATH", "/opt/models/gliner2.5-decide"),
            device=os.environ.get("DEVICE", "cuda"),
            precision=os.environ.get("PRECISION"),
            idle_timeout=runtime.initial_worker_idle_timeout(),
        )
        admin_config = load_admin_config()
        servers = build_servers(runtime=runtime, admin_config=admin_config)
    except Exception as exc:
        logging.error("Configuration error: %s", exc)
        sys.exit(1)

    if admin_config.enabled:
        logging.info(
            "Internal admin API configured: bind=%s:%d",
            servers[1].config.host,
            servers[1].config.port,
        )
    else:
        logging.info("Internal admin API disabled (ADMIN_API_TOKEN not set)")

    asyncio.run(serve_servers(servers))


if __name__ == "__main__":
    main()
