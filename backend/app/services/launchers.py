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


class SubprocessSimulationLauncher:
    def __init__(self, settings: Settings, sessions: sessionmaker[Session]) -> None:
        self._settings = settings
        self._sessions = sessions
        self._log = get_logger("backend.launcher")

    def launch(self, experiment_id: str) -> None:
        env = {
            **os.environ,
            "FEDRAD_DATABASE_URL": self._settings.database_url,
            "FEDRAD_MLFLOW_TRACKING_URI": self._settings.mlflow_tracking_uri,
            "FEDRAD_MLFLOW_ARTIFACT_ROOT": self._settings.mlflow_artifact_root,
        }
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "experiments.run_simulation", "--experiment-id", experiment_id],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self._log.info("simulation_finished", experiment_id=experiment_id, rc=result.returncode)
        if result.returncode != 0:
            with self._sessions() as session:
                experiment_service.mark_failed(
                    session, experiment_id, result.stderr or "simulation process failed"
                )
