"""Backend API, persistence and tracking, driven by the simulated federation."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.services.recording import DatabaseRunRecorder
from backend.app.settings import Settings
from backend.app.tracking import make_tracker
from experiments.simulation import build_clients, run_simulation
from shared.config import ExperimentConfig


class InlineSimulationLauncher:
    """Test launcher: runs the simulation in-process instead of a subprocess."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.sessions = None

    def launch(self, experiment_id: str) -> None:
        from backend.app.services import experiments as service

        with self.sessions() as session:
            config = ExperimentConfig.model_validate(
                service.get_experiment(session, experiment_id).config
            )
        tracker = make_tracker(
            self._settings.mlflow_tracking_uri, self._settings.mlflow_artifact_root
        )
        run_simulation(
            experiment_id, config,
            recorder=DatabaseRunRecorder(experiment_id, self.sessions, tracker),
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path}/app.db",
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_artifact_root=str(tmp_path / "artifacts"),
    )


@pytest.fixture
def api(settings) -> TestClient:
    launcher = InlineSimulationLauncher(settings)
    app = create_app(settings, launcher=launcher)
    launcher.sessions = app.state.sessions
    with TestClient(app) as client:
        yield client


@pytest.fixture
def finished(api, config) -> str:
    created = api.post("/experiments", json=config.model_dump(mode="json"))
    assert created.status_code == 201, created.text
    experiment_id = created.json()["id"]
    assert created.json()["status"] == "created"
    assert api.post(f"/experiments/{experiment_id}/run").status_code == 202
    return experiment_id  # TestClient runs background tasks before returning


def test_experiment_lifecycle(api, finished):
    experiment = api.get(f"/experiments/{finished}").json()
    assert experiment["status"] == "completed"
    assert experiment["data_origin"] == "synthetic"
    assert experiment["mlflow_run_id"]
    assert experiment["config"]["seed"] == 20261116
    assert [e["id"] for e in api.get("/experiments").json()] == [finished]
    # An experiment runs at most once.
    assert api.post(f"/experiments/{finished}/run").status_code == 409


def test_clients_rounds_results_and_costs(api, finished):
    clients = api.get("/clients", params={"experiment_id": finished}).json()
    assert [(c["client_id"], c["status"]) for c in clients] == [
        ("client-01", "registered"), ("client-02", "registered"), ("client-03", "registered"),
    ]
    rounds = api.get("/federated-rounds", params={"experiment_id": finished}).json()
    assert [(r["round_number"], r["task"], r["status"]) for r in rounds] == [
        (1, "feature_inventory", "completed"), (2, "feature_summary", "completed"),
    ]
    detail = api.get(f"/federated-rounds/{rounds[1]['id']}").json()
    assert {p["client_id"]: p["n_samples"] for p in detail["participations"]} == {
        "client-01": 60, "client-02": 45, "client-03": 80,
    }
    results = api.get(f"/results/{finished}").json()
    assert results["data_origin"] == "synthetic"
    assert results["rounds"][1]["aggregate"]["n_samples"] == 185
    costs = api.get(f"/results/{finished}/costs").json()
    metrics = {c["metric"] for c in costs}
    assert {
        "communication_time", "aggregation_time", "bytes_sent", "bytes_received",
        "local_execution_time", "peak_memory", "total_bytes", "n_rounds",
    } <= metrics
    assert api.get("/models", params={"experiment_id": finished}).json() == []


def test_invalid_configuration_and_unknown_resources(api, config):
    bad = {**config.model_dump(mode="json"), "expected_clients": ["client-01"]}
    assert api.post("/experiments", json=bad).status_code == 422
    assert api.get("/experiments/exp-missing").status_code == 404
    assert api.post("/experiments/exp-missing/run").status_code == 404
    assert api.get("/results/exp-missing").status_code == 404
    assert api.get("/federated-rounds/nope").status_code == 404
    assert api.get("/health").json() == {"status": "ok"}


def test_central_database_holds_no_individual_data(api, finished, settings, config):
    """Scan every stored value against the clients' local data."""
    db_path = settings.database_url.removeprefix("sqlite:///")
    with sqlite3.connect(db_path) as connection:
        tables = [r[0] for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        dump = json.dumps(
            {t: connection.execute(f"SELECT * FROM {t}").fetchall() for t in tables}, default=str
        )
        columns = {
            row[1] for t in tables for row in connection.execute(f"PRAGMA table_info({t})")
        }
    assert {"experiments", "clients", "federated_rounds", "cost_records"} <= set(tables)
    assert columns.isdisjoint({"participant_id", "time_months", "event", "image", "mask"})
    assert "SYN-" not in dump
    for client in build_clients(finished, config):
        dataset = client._data_source.load()
        for value in dataset.features[:5].ravel().tolist() + dataset.time_months[:5].tolist():
            assert repr(value) not in dump
    # Only the global aggregate is stored: no per-site means.
    assert dump.count('\\"mean\\"') + dump.count('"mean"') == 1


def test_run_is_tracked_in_mlflow(api, finished, settings):
    from mlflow.tracking import MlflowClient

    run_id = api.get(f"/experiments/{finished}").json()["mlflow_run_id"]
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["data_origin"] == "synthetic"
    assert run.data.params["seed"] == "20261116"
    assert run.data.metrics["n_rounds"] == 2
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert {"experiment_config.json", "aggregates"} <= artifacts


def test_subprocess_launcher_end_to_end(settings, config):
    """Default wiring: the API spawns the simulation as a separate process."""
    with TestClient(create_app(settings)) as api:
        experiment_id = api.post("/experiments", json=config.model_dump(mode="json")).json()["id"]
        assert api.post(f"/experiments/{experiment_id}/run").status_code == 202
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            experiment = api.get(f"/experiments/{experiment_id}").json()
            if experiment["status"] in {"completed", "failed"}:
                break
            time.sleep(0.2)
        assert experiment["status"] == "completed", experiment["error"]
        assert len(api.get(f"/results/{experiment_id}").json()["rounds"]) == 2
