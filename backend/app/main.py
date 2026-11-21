"""FastAPI application factory."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.db import init_db, make_engine
from backend.app.routers import (
    clients,
    experiments,
    federated_rounds,
    models,
    reports,
    results,
)
from backend.app.security import require_token
from backend.app.services.launchers import ExperimentLauncher, make_launcher
from backend.app.settings import Settings, get_settings
from shared.logging import configure_logging

_UNSET = object()


def create_app(
    settings: Settings | None = None, launcher: ExperimentLauncher | None | object = _UNSET
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="Federated Radiomics API",
        version="0.7.0",
        description="Experiment management for federated radiomics (MCI -> AD conversion).",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
        allow_headers=["*"],
    )
    sessions = init_db(make_engine(settings.database_url))
    app.state.sessions = sessions
    app.state.settings = settings
    if launcher is _UNSET:
        launcher = make_launcher(settings, sessions)
    app.state.launcher = launcher

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    for module in (experiments, clients, federated_rounds, models, results, reports):
        app.include_router(module.router, dependencies=[Depends(require_token)])
    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn backend.app.main:app_factory --factory``."""
    return create_app()
