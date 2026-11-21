"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from backend.app.services.launchers import ExperimentLauncher


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.sessions() as session:
        yield session


def get_launcher(request: Request) -> ExperimentLauncher | None:
    return request.app.state.launcher
