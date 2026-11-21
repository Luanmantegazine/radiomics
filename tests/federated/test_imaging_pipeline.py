"""Local extraction pipeline, its outputs, and their use in the federation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from client.feature_table import FeatureTableDataSource, FeatureTableError
from client.imaging.__main__ import main as imaging_cli
from client.imaging.pipeline import run_extraction
from coordinator.transport import Direction, InProcessTransport
from experiments.simulation import run_simulation
from shared.config import ExperimentConfig
from shared.protocol import DataOrigin
from tests.federated import phantoms


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """Five participants x two ROIs; one has no follow-up, one has a bad mask."""
    root = tmp_path_factory.mktemp("site")
    manifest = phantoms.write_site(root, 5, seed=7)
    rows = list(pd.read_csv(manifest, dtype=str).fillna("").to_dict("records"))
    for row in rows:
        if row["participant_id"] == "P07001":
            row["time_months"], row["event"] = "", ""
        if row["participant_id"] == "P07003" and row["roi"] == "sphere":
            row["mask_label"] = "9"  # label absent from the mask
    phantoms.write_manifest(manifest, rows)
    output = root / "out"
    result = run_extraction(manifest, phantoms.config(), output)
    return root, manifest, output, result


def test_outputs(site):
    _, _, output, result = site
    assert (result.n_cases, result.n_extracted) == (10, 9)
    assert (result.n_participants, result.n_complete_participants) == (5, 4)
    table = pd.read_csv(output / "features.csv", dtype={"participant_id": str})
    assert list(table["participant_id"]) == ["P07000", "P07001", "P07002", "P07004"]
    assert table.shape == (4, 1 + 2 * 107)
    assert table.columns[1] == "box__original_shape_Elongation"
    assert "sphere__original_firstorder_Mean" in table.columns
    assert np.isfinite(table.drop(columns="participant_id").to_numpy()).all()

    cases = pd.read_csv(output / "cases.csv", dtype=str).fillna("")
    bad = cases[(cases.participant_id == "P07003") & (cases.roi == "sphere")].iloc[0]
    assert (bad.status, bad.errors) == ("invalid", "label_absent")
    good = cases[cases.status == "extracted"].iloc[0]
    assert good.image_hash and good.mask_hash and good.spacing == "1x1x1"


def test_provenance(site):
    _, manifest, output, result = site
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["config_id"] == "t1-original-v1"
    assert provenance["features_sha256"] == result.features_sha256
    assert provenance["rois"] == ["box", "sphere"]
    assert len(provenance["feature_names"]) == 214
    assert provenance["counts"] == {
        "cases": 10, "cases_extracted": 9, "cases_invalid": 1, "cases_failed": 0,
        "participants": 5, "participants_complete": 4,
        "participants_with_followup": 4, "participants_without_followup": 1,
    }
    run_info = json.loads((output / "run_info.json").read_text())
    assert run_info["extraction_time_s"] > 0 and run_info["peak_memory_bytes"] > 0


def test_results_are_reproducible(site, tmp_path):
    _, manifest, output, _ = site
    run_extraction(manifest, phantoms.config(), tmp_path / "again")
    for name in ("features.csv", "cases.csv", "provenance.json"):
        assert (output / name).read_bytes() == (tmp_path / "again" / name).read_bytes(), name


def test_existing_results_are_not_overwritten_silently(site):
    _, manifest, output, _ = site
    with pytest.raises(FileExistsError):
        run_extraction(manifest, phantoms.config(), output)


def test_cli(site, tmp_path, capsys):
    _, manifest, _, _ = site
    config = str(phantoms.DEFAULT_CONFIG)
    assert imaging_cli(["check", "--manifest", str(manifest), "--config", config,
                        "--log-level", "ERROR"]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["cases_ok"] == 9 and summary["issues"] == {"error:label_absent": 1}
    assert summary["followup"]["n_without_followup"] == 1
    assert imaging_cli(["extract", "--manifest", str(manifest), "--config", config,
                        "--output-dir", str(tmp_path / "cli"), "--log-level", "ERROR"]) == 1
    assert json.loads(capsys.readouterr().out)["participants_complete"] == 4


def test_feature_table_source_excludes_unknown_followup(site):
    _, manifest, output, _ = site
    dataset = FeatureTableDataSource(
        output / "features.csv", manifest, DataOrigin.SYNTHETIC
    ).load()
    # P07001 has features but no follow-up: excluded, not treated as censored.
    assert dataset.participant_ids == ("P07000", "P07002", "P07004")
    assert dataset.features.shape == (3, 214)
    assert dataset.event.tolist() == [True, False, False]
    assert (dataset.time_months > 0).all()


def test_modified_table_is_rejected(site, tmp_path):
    _, manifest, output, _ = site
    for name in ("features.csv", "provenance.json"):
        (tmp_path / name).write_bytes((output / name).read_bytes())
    with (tmp_path / "features.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(FeatureTableError, match="provenance"):
        FeatureTableDataSource(tmp_path / "features.csv", manifest, DataOrigin.SYNTHETIC).load()


def test_federation_over_extracted_features(tmp_path):
    """Two phantom sites: extract locally, then run the Stage 1 flow on the tables."""
    clients = []
    for index, seed in enumerate((11, 12), start=1):
        root = tmp_path / f"site{index}"
        manifest = phantoms.write_site(root, 4, seed=seed, rois={"box": 1})
        run_extraction(manifest, phantoms.config(), root / "out")
        clients.append(
            {
                "client_id": f"client-0{index}",
                "data_source": {
                    "kind": "feature_table", "features_path": str(root / "out/features.csv"),
                    "manifest_path": str(manifest), "data_origin": "synthetic",
                },
            }
        )
    config = ExperimentConfig.model_validate(
        {
            "name": "phantom-feature-tables", "seed": 1, "data_origin": "synthetic",
            "privacy": {"min_cohort_size": 4, "min_clients_per_aggregate": 2},
            "expected_clients": ["client-01", "client-02"],
            "rounds": [
                {"phase": "infrastructure", "task": "feature_inventory",
                 "aggregator": "schema_intersection"},
                {"phase": "infrastructure", "task": "feature_summary",
                 "aggregator": "pooled_moments"},
            ],
            "simulation": {"clients": clients},
        }
    )
    transport = InProcessTransport(capture=True)
    outcome = run_simulation("exp-phantom", config, transport=transport)
    assert outcome.completed
    summary = outcome.rounds[1].aggregate
    assert summary.n_samples == 8 and len(summary.feature_names) == 107

    tables = [pd.read_csv(c["data_source"]["features_path"]) for c in clients]
    pooled = pd.concat(tables)[list(summary.feature_names)].to_numpy()
    np.testing.assert_allclose(summary.mean, pooled.mean(axis=0), rtol=1e-9)

    # Isolation still holds with the imaging-backed source.
    wire = b"\n".join(
        r.raw for r in transport.log.records if r.direction is Direction.TO_COORDINATOR
    ).decode()
    for table in tables:
        assert not any(pid in wire for pid in table["participant_id"])
        # (shape features are identical across phantoms, so use an intensity feature)
        assert repr(float(table["box__original_firstorder_Mean"].iloc[0])) not in wire


def test_real_and_synthetic_sources_cannot_be_mixed(config):
    raw = config.model_dump(mode="json")
    raw["simulation"]["clients"][0]["data_source"] = {
        "kind": "feature_table", "features_path": "f.csv", "manifest_path": "m.csv",
        "data_origin": "real",
    }
    with pytest.raises(ValueError, match="data_origin"):
        ExperimentConfig.model_validate(raw)
