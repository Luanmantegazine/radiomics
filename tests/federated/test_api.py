"""Stage 7: consolidated API - migrations, auth, templates, summaries, reports."""

from __future__ import annotations

import json

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from backend.app.db import make_engine
from backend.app.main import create_app
from backend.app.migrations import UnversionedDatabase, current_revision, upgrade
from backend.app.models import Base
from backend.app.settings import Settings
from shared.config import load_experiment_config
from tests.federated.test_backend import InlineSimulationLauncher
from tests.federated.test_harmonization import STAGE6


def _settings(tmp_path, **changes) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path}/app.db",
        mlflow_tracking_uri="none",
        **changes,
    )


def _client(settings) -> TestClient:
    launcher = InlineSimulationLauncher(settings)
    app = create_app(settings, launcher=launcher)
    launcher.sessions = app.state.sessions
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Migrations
# --------------------------------------------------------------------------- #
def test_migrations_create_the_schema_of_the_models(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/m.db")
    upgrade(engine)
    assert current_revision(engine) == "0001"
    with engine.connect() as connection:
        drift = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    assert drift == [], "models changed without a migration: run `python -m " \
                        "backend.app.migrations revision -m <message>`"
    upgrade(engine)  # idempotent


def test_unversioned_databases_are_not_touched(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/old.db")
    Base.metadata.create_all(engine)  # how Stages 1-6 created development databases
    with pytest.raises(UnversionedDatabase, match="no migration history"):
        upgrade(engine)


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def test_optional_bearer_token(tmp_path):
    with _client(_settings(tmp_path, api_token="s3cret")) as api:
        assert api.get("/health").status_code == 200
        assert api.get("/experiments").status_code == 401
        assert api.get("/experiments", headers={"Authorization": "Bearer nope"}).status_code == 401
        ok = api.get("/experiments", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200 and ok.json() == []
        assert api.get("/reports/x", headers={"Authorization": "Basic s3cret"}).status_code == 401
    with _client(_settings(tmp_path / "open")) as api:
        assert api.get("/experiments").status_code == 200


# --------------------------------------------------------------------------- #
# Templates and validation
# --------------------------------------------------------------------------- #
def test_templates_and_validation(tmp_path):
    with _client(_settings(tmp_path)) as api:
        templates = {t["id"]: t for t in api.get("/experiments/templates").json()}
        assert set(templates) == {
            "stage1_simulation", "stage4_federated_selection", "stage5_federated_models",
            "stage6_harmonization_evaluation",
        }  # extraction/reliability/selection configs in subdirectories are not offered
        stage6 = templates["stage6_harmonization_evaluation"]
        assert stage6["phases"] == ["harmonization", "feature_selection", "training",
                                    "evaluation"]
        result = api.post("/experiments/validate", json=stage6["config"]).json()
        assert result["valid"] and result["data_origin"] == "synthetic"
        created = api.post("/experiments", json=stage6["config"]).json()
        assert created["config_hash"] == result["config_hash"]

        broken = json.loads(json.dumps(stage6["config"]))
        broken["evaluation"]["decision_thresholds"] = [0.15]
        broken["seed"] = -1
        result = api.post("/experiments/validate", json=broken).json()
        assert not result["valid"]
        locations = {e["location"] for e in result["errors"]}
        assert "seed" in locations
        assert api.post("/experiments/validate", json={"name": "x"}).json()["valid"] is False


# --------------------------------------------------------------------------- #
# Summaries and reports on a full run
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    settings = _settings(tmp)
    with _client(settings) as api:
        config = load_experiment_config(STAGE6).model_dump(mode="json")
        experiment_id = api.post("/experiments", json=config).json()["id"]
        api.post(f"/experiments/{experiment_id}/run")
        yield api, experiment_id, settings


def test_summary(full_run):
    api, experiment_id, _ = full_run
    summary = api.get(f"/experiments/{experiment_id}/summary").json()
    assert summary["experiment"]["status"] == "completed"
    assert summary["registered_clients"] == 3 and summary["rejected_clients"] == 0
    assert set(summary["rounds"]["by_phase"]) == {"harmonization", "feature_selection",
                                                   "training", "evaluation"}
    assert summary["rounds"]["failed"] == 0
    assert all(summary["available_results"].values())
    assert summary["total_bytes"] > 0
    assert api.get("/experiments/nope/summary").status_code == 404
    listed = api.get("/experiments", params={"status": "completed", "limit": 1}).json()
    assert [e["id"] for e in listed] == [experiment_id]
    assert api.get("/experiments", params={"status": "failed"}).json() == []


def test_cost_summary_is_consistent(full_run):
    api, experiment_id, _ = full_run
    costs = api.get(f"/results/{experiment_id}/costs/summary").json()
    per_round = sum(r["bytes_from_clients"] + r["bytes_to_clients"] for r in costs["by_round"])
    assert per_round == pytest.approx(costs["totals"]["total_bytes"])
    by_phase = sum(p["bytes_from_clients"] + p["bytes_to_clients"] for p in costs["by_phase"])
    assert by_phase == pytest.approx(per_round)
    assert sum(p["rounds"] for p in costs["by_phase"]) == len(costs["by_round"])
    clients = {c["client_id"]: c for c in costs["by_client"]}
    assert set(clients) == {"client-01", "client-02", "client-03"}
    sent = sum(c["bytes_sent"] for c in clients.values())
    assert sent == pytest.approx(sum(r["bytes_from_clients"] for r in costs["by_round"]))
    assert all(c["peak_memory_bytes"] > 0 for c in clients.values())


def test_typed_result_endpoints(full_run):
    api, experiment_id, _ = full_run
    schema = api.get("/openapi.json").json()["paths"]
    for path in ("/results/{experiment_id}/feature-selection",
                 "/results/{experiment_id}/harmonization",
                 "/results/{experiment_id}/evaluation",
                 "/results/{experiment_id}/costs/summary"):
        content = schema[path]["get"]["responses"]["200"]["content"]["application/json"]
        assert "$ref" in content["schema"], path
    evaluation = api.get(f"/results/{experiment_id}/evaluation").json()
    assert evaluation["evaluations"][0]["pooled"][0]["available"] is True


def test_reports(full_run):
    api, experiment_id, _ = full_run
    bundle = api.get(f"/reports/{experiment_id}").json()
    assert set(bundle) >= {"summary", "clients", "rounds", "harmonization",
                           "feature_selection", "models", "evaluation", "costs", "notice"}
    assert "SYNTHETIC DATA" in bundle["notice"]
    assert bundle["summary"]["experiment"]["config"]["seed"] == 20261118

    html = api.get(f"/reports/{experiment_id}", params={"format": "html"})
    assert html.headers["content-type"].startswith("text/html")
    page = html.text
    assert "DADOS SINTÉTICOS" in page and "stage6-harmonization-evaluation" in page
    assert "<script" not in page  # self-contained, static
    download = api.get(f"/reports/{experiment_id}",
                       params={"format": "html", "download": True})
    assert "attachment" in download.headers["content-disposition"]

    for table in ("evaluation", "selected-features", "models", "costs", "rounds"):
        response = api.get(f"/reports/{experiment_id}/{table}.csv")
        assert response.status_code == 200, table
        lines = response.text.splitlines()
        assert lines[0].startswith("# SYNTHETIC DATA") and len(lines) > 2, table
    assert api.get(f"/reports/{experiment_id}/secrets.csv").status_code == 404
    assert api.get("/reports/nope").status_code == 404

    # Reports are built from released results only: no participant identifier.
    for text in (json.dumps(bundle), page):
        assert "SYN-client" not in text


def test_dashboard_types_match_the_api(tmp_path):
    """frontend/openapi.json (source of the TypeScript types) must be current.

    Regenerate with: python -m backend.app.openapi frontend/openapi.json &&
    (cd frontend && npm run gen:api)
    """
    from pathlib import Path

    from backend.app.openapi import main as write_openapi

    current = tmp_path / "openapi.json"
    write_openapi([str(current)])
    committed = Path(__file__).resolve().parents[2] / "frontend" / "openapi.json"
    assert json.loads(current.read_text()) == json.loads(committed.read_text())
