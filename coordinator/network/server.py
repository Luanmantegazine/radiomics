"""Run the site endpoints in a background thread (uvicorn)."""

from __future__ import annotations

import threading
import time

import uvicorn
from fastapi import FastAPI


class BackgroundServer:
    def __init__(self, app: FastAPI, host: str, port: int, *, ssl_certfile: str | None = None,
                 ssl_keyfile: str | None = None) -> None:
        self._server = uvicorn.Server(uvicorn.Config(
            app, host=host, port=port, log_level="warning", ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile, timeout_keep_alive=75,
        ))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def port(self) -> int:
        servers = getattr(self._server, "servers", None) or []
        for server in servers:
            for sock in server.sockets:
                return sock.getsockname()[1]
        return self._server.config.port

    def start(self, timeout_s: float = 10.0) -> BackgroundServer:
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("site endpoint server did not start")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
