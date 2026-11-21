"""Reproducibility: digests of released results, provenance, migrations."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.db import make_engine
from backend.app.main import create_app
from backend.app.migrations import current_revision, upgrade
from backend.app.settings import Settings
from coordinator.provenance import code_provenance, outcome_digest, released_results
from experiments.reproduce import main as reproduce
from experiments.simulation import run_simulation
from shared.config import ExperimentConfig, load_experiment_config
from tests.federated.conftest import CONFIG_PATH
from tests.federated.test_backend import InlineSimulationLauncher

STAGE5 = CONFIG_PATH.parent / "stage5_federated_models.yaml"


def test_digest_is_stable_and_sensitive_to_the_configuration():
    config = load_experiment_config(STAGE5)
    first, second = run_simulation("exp-r", config), run_simulation("exp-r", config)
    assert first.duration_s != second.duration_s  # timings differ ...
    assert outcome_digest(first) == outcome_digest(second)  # ... released results do not
    other = ExperimentConfig.model_validate({**config.model_dump(mode="json"), "seed": 1})
    assert outcome_digest(run_simulation("exp-r", other)) != outcome_digest(first)
    released = json.dumps(released_results(first))
    assert "round_trip" not in released and "duration" not in released


def test_code_provenance():
    provenance = code_provenance()
    assert len(provenance["uv_lock_sha256"]) == 64
    assert provenance["git_commit"] is None or len(provenance["git_commit"]) == 40
    assert provenance["libraries"]["numpy"].startswith("1.26.")


def test_reproduce_cli_on_configs(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    assert reproduce(["--config", str(CONFIG_PATH), "--output", str(manifest_path)]) == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["reproducible"] and len(set(manifest["digests"])) == 1
    capsys.readouterr()


def test_recorded_experiments_carry_digest_and_can_be_checked(tmp_path, monkeypatch, capsys):
    settings = Settings(database_url=f"sqlite:///{tmp_path}/r.db", mlflow_tracking_uri="none")
    launcher = InlineSimulationLauncher(settings)
    app = create_app(settings, launcher=launcher)
    launcher.sessions = app.state.sessions
    with TestClient(app) as api:
        config = load_experiment_config(STAGE5).model_dump(mode="json")
        experiment_id = api.post("/experiments", json=config).json()["id"]
        api.post(f"/experiments/{experiment_id}/run")
        stored = api.get(f"/experiments/{experiment_id}").json()
    assert len(stored["outcome_digest"]) == 64
    assert stored["code_version"]["uv_lock_sha256"] == code_provenance()["uv_lock_sha256"]

    monkeypatch.setenv("FEDRAD_DATABASE_URL", settings.database_url)
    from backend.app.settings import get_settings

    get_settings.cache_clear()
    try:
        assert reproduce(["--experiment-id", experiment_id]) == 0
        assert json.loads(capsys.readouterr().out)["reproducible"] is True
        engine = make_engine(settings.database_url)
        with engine.begin() as connection:
            connection.execute(text("UPDATE experiments SET outcome_digest = :d"),
                               {"d": "0" * 64})
        assert reproduce(["--experiment-id", experiment_id]) == 1
        report = json.loads(capsys.readouterr().out)
        assert report["reproducible"] is False and report["code_differences"] == {}
    finally:
        get_settings.cache_clear()


def test_migration_0002_keeps_existing_rows(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/old.db")
    upgrade(engine, "0001")
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO experiments (id, name, status, data_origin, protocol_version, seed, "
            "config, config_hash, created_at) VALUES ('exp-old', 'old', 'completed', "
            "'synthetic', '1.3', 1, '{}', 'abc', '2026-01-01')"))
    upgrade(engine)
    assert current_revision(engine) == "0002"
    with engine.connect() as connection:
        row = connection.execute(text(
            "SELECT id, outcome_digest, code_version FROM experiments")).one()
    assert row == ("exp-old", None, None)
