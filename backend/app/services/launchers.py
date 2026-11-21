"""Launchers: how the backend triggers a federated run.

The backend never runs federated or scientific code in its own process. In
Stage 1 it spawns the single-machine simulation as a separate process; a
distributed deployment replaces this with a call to the coordinator service.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Protocol

from sqlalchemy.orm import Session, sessionmaker

from backend.app.services import experiments as experiment_service
from backend.app.settings import Settings
from shared.logging import get_logger


class ExperimentLauncher(Protocol):
    def launch(self, experiment_id: str) -> None:
        """Run the experiment to completion (called from a background task)."""
        ...


class SubprocessLauncher:
    """Runs a federated run in a separate process and marks failures."""

    module = "experiments.run_simulation"

    def __init__(self, settings: Settings, sessions: sessionmaker[Session]) -> None:
        self._settings = settings
        self._sessions = sessions
        self._log = get_logger("backend.launcher")

    def extra_args(self) -> list[str]:
        return []

    def launch(self, experiment_id: str) -> None:
        env = {
            **os.environ,
            "FEDRAD_DATABASE_URL": self._settings.database_url,
            "FEDRAD_MLFLOW_TRACKING_URI": self._settings.mlflow_tracking_uri,
            "FEDRAD_MLFLOW_ARTIFACT_ROOT": self._settings.mlflow_artifact_root,
        }
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", self.module, "--experiment-id", experiment_id,
             *self.extra_args()],
            env=env, capture_output=True, text=True, check=False,
        )
        self._log.info("run_finished", experiment_id=experiment_id, rc=result.returncode,
                       module=self.module)
        if result.returncode != 0:
            with self._sessions() as session:
                experiment_service.mark_failed(
                    session, experiment_id, result.stderr or "federated run failed"
                )


class SubprocessDistributedLauncher(SubprocessLauncher):
    """Starts the coordinator; remote sites connect to it (Stage 8)."""

    module = "experiments.run_distributed"

    def extra_args(self) -> list[str]:
        s = self._settings
        if not s.sites_file:
            raise ValueError("FEDRAD_SITES_FILE is required by the distributed launcher")
        return ["--sites", s.sites_file, "--port", str(s.coordinator_port),
                "--registration-timeout", str(s.registration_timeout_s),
                "--round-timeout", str(s.round_timeout_s)]


class SubprocessSimulationLauncher(SubprocessLauncher):
    """Single-machine simulation (Stage 1)."""

    module = "experiments.run_simulation"


def make_launcher(settings: Settings, sessions: sessionmaker[Session]):
    if settings.launcher == "simulation":
        return SubprocessSimulationLauncher(settings, sessions)
    if settings.launcher == "distributed":
        return SubprocessDistributedLauncher(settings, sessions)
    return None
