"""Experiment tracking port and its MLflow implementation."""

from __future__ import annotations

import os
from typing import Any, Protocol


class ExperimentTracker(Protocol):
    def start_run(self, experiment_name: str, run_name: str, params: dict[str, Any],
                  tags: dict[str, str]) -> str | None: ...

    def log_metrics(self, run_id: str | None, metrics: dict[str, float], step: int) -> None: ...

    def log_dict(self, run_id: str | None, data: dict[str, Any], artifact_file: str) -> None: ...

    def set_tags(self, run_id: str | None, tags: dict[str, str]) -> None: ...

    def end_run(self, run_id: str | None, succeeded: bool) -> None: ...


class NullTracker:
    """Tracking disabled."""

    def start_run(self, experiment_name, run_name, params, tags) -> str | None:
        return None

    def log_metrics(self, run_id, metrics, step) -> None:
        return None

    def log_dict(self, run_id, data, artifact_file) -> None:
        return None

    def set_tags(self, run_id, tags) -> None:
        return None

    def end_run(self, run_id, succeeded) -> None:
        return None


class MlflowTracker:
    """Tracks runs with MLflow using explicit run ids (no global active run)."""

    def __init__(self, tracking_uri: str, artifact_root: str = "") -> None:
        # MLflow sends usage telemetry to an external host by default; a medical
        # research deployment should not. An explicit setting still wins.
        os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
        os.environ.setdefault("DO_NOT_TRACK", "true")
        from mlflow.tracking import MlflowClient  # imported lazily: heavy dependency

        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._artifact_root = artifact_root

    def start_run(self, experiment_name, run_name, params, tags) -> str | None:
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
        else:
            location = f"{self._artifact_root.rstrip('/')}/{experiment_name}" \
                if self._artifact_root else None
            experiment_id = self._client.create_experiment(
                experiment_name, artifact_location=location
            )
        run = self._client.create_run(experiment_id, run_name=run_name, tags=tags)
        for key, value in params.items():
            self._client.log_param(run.info.run_id, key, value)
        return run.info.run_id

    def log_metrics(self, run_id, metrics, step) -> None:
        for key, value in metrics.items():
            self._client.log_metric(run_id, key, float(value), step=step)

    def log_dict(self, run_id, data, artifact_file) -> None:
        self._client.log_dict(run_id, data, artifact_file)

    def set_tags(self, run_id, tags) -> None:
        for key, value in tags.items():
            self._client.set_tag(run_id, key, value)

    def end_run(self, run_id, succeeded) -> None:
        self._client.set_terminated(run_id, status="FINISHED" if succeeded else "FAILED")


def make_tracker(tracking_uri: str, artifact_root: str = "") -> ExperimentTracker:
    if tracking_uri.lower() in {"", "none"}:
        return NullTracker()
    return MlflowTracker(tracking_uri, artifact_root)
