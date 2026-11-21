"""Local selectors, reference comparators and the comparison study."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from client.data import LocalDataset
from client.imaging.pipeline import run_extraction
from client.selection.__main__ import main as selection_cli
from client.selection.config import SelectionStudyConfig, load_selection_config
from client.selection.selectors import SelectionError, SelectionInputs, run_selector
from client.selection.study import run_selection_study, write_study
from shared.protocol import DataOrigin
from tests.federated import phantoms

SHIPPED = phantoms.REPO / "experiments/configs/selection/local_v1.yaml"


def _config(n_features=3, min_events=10, bootstrap=None, reliability="default"):
    raw = load_selection_config(SHIPPED).model_dump(mode="json")
    raw["min_events"], raw["bootstrap"] = min_events, bootstrap
    for selector in raw["selectors"]:
        selector["n_features"] = n_features
    if reliability != "default":
        raw["selectors"][0]["reliability"] = reliability
    return SelectionStudyConfig.model_validate(raw)


def _spec(config, method):
    return next(s for s in config.selectors if s.method == method)


@pytest.fixture(scope="module")
def cohort():
    names, x, time, event = phantoms.survival_cohort(seed=11)
    return SelectionInputs(names, x, time, event)


def _with_reliability(inputs, values):
    table = {name: 0.95 for name in inputs.feature_names} | values
    return SelectionInputs(
        inputs.feature_names, inputs.features, inputs.time, inputs.event, {"min_icc": table}
    )


def test_shipped_configuration_has_no_implicit_thresholds():
    config = load_selection_config(SHIPPED)
    assert [s.method for s in config.selectors] == [
        "component", "correlation_filter", "lasso_cox", "graph_representatives",
    ]
    raw = config.model_dump(mode="json")
    del raw["selectors"][0]["redundancy"]  # must be given, even if null
    with pytest.raises(ValidationError):
        SelectionStudyConfig.model_validate(raw)
    raw = config.model_dump(mode="json")
    del raw["min_events"]
    with pytest.raises(ValidationError):
        SelectionStudyConfig.model_validate(raw)


def test_component_selector_combines_the_three_dimensions(cohort):
    spec = _spec(_config(), "component")
    # f0 is the strongest predictor but is declared unreliable; f2 is its near-copy.
    inputs = _with_reliability(cohort, {"f0": 0.40, "f7": float("nan")})
    result = run_selector(spec, inputs, min_events=10)
    assert result.excluded == {
        "f0": "below_reliability_threshold", "f7": "reliability_unknown",
    }
    assert result.selected[:2] in (("f2", "f1"), ("f1", "f2"))
    assert len(result.selected) == 3 and "f0" not in result.selected

    # With every feature reliable, f0 and f2 are redundant: only one survives.
    result = run_selector(spec, _with_reliability(cohort, {}), min_events=10)
    assert len({"f0", "f2"} & set(result.selected)) == 1
    assert "f1" in result.selected
    assert list(result.scores) == list(result.selected)


def test_component_selector_never_skips_reliability_silently(cohort):
    config = _config()
    with pytest.raises(SelectionError, match="no reliability estimates were provided"):
        run_selector(_spec(config, "component"), cohort, min_events=10)
    # Disabling the reliability step must be an explicit configuration choice.
    explicit = _spec(_config(reliability=None), "component")
    assert len(run_selector(explicit, cohort, min_events=10).selected) == 3


def test_redundancy_step_can_be_disabled(cohort):
    raw = _config(reliability=None).model_dump(mode="json")
    raw["selectors"][0]["redundancy"] = None
    spec = SelectionStudyConfig.model_validate(raw).selectors[0]
    assert {"f0", "f2"} <= set(run_selector(spec, cohort, min_events=10).selected)


def test_reference_selectors(cohort):
    config = _config()
    filtered = run_selector(_spec(config, "correlation_filter"), cohort, min_events=10)
    assert len({"f0", "f2"} & set(filtered.selected)) == 1 and "f1" in filtered.selected

    lasso = run_selector(_spec(config, "lasso_cox"), cohort, min_events=10)
    assert 1 <= len(lasso.selected) <= 3
    assert "f1" in lasso.selected and {"f0", "f2"} & set(lasso.selected)
    assert all(score > 0 for score in lasso.scores.values())

    graph = run_selector(_spec(config, "graph_representatives"), cohort, min_events=10)
    assert len({"f0", "f2"} & set(graph.selected)) == 1 and "f1" in graph.selected


def test_constant_and_non_finite_features_are_excluded(cohort):
    x = cohort.features.copy()
    x[:, 3], x[0, 4] = 5.0, np.nan
    inputs = SelectionInputs(cohort.feature_names, x, cohort.time, cohort.event)
    for method in ("correlation_filter", "lasso_cox", "graph_representatives"):
        result = run_selector(_spec(_config(n_features=12), method), inputs, min_events=10)
        assert result.excluded == {"f3": "constant", "f4": "non_finite"}, method
        assert not {"f3", "f4"} & set(result.selected)


def test_selectors_refuse_to_run_with_too_few_events(cohort):
    config = _config()
    for spec in config.selectors[1:]:
        with pytest.raises(SelectionError, match="events observed"):
            run_selector(spec, cohort, min_events=10_000)


def _dataset(inputs) -> LocalDataset:
    return LocalDataset(
        participant_ids=tuple(f"S{i}" for i in range(len(inputs.time))),
        feature_names=inputs.feature_names,
        features=inputs.features,
        time_months=inputs.time,
        event=inputs.event,
        origin=DataOrigin.SYNTHETIC,
    )


def test_study_compares_selectors_and_measures_stability(cohort, tmp_path):
    config = _config(bootstrap={"n_resamples": 20, "seed": 3})
    reliability = {"min_icc": {name: 0.9 for name in cohort.feature_names}}
    result = run_selection_study(_dataset(cohort), config, reliability)
    again = run_selection_study(_dataset(cohort), config, reliability)
    assert result == again  # seeded bootstrap

    names = [s.name for s in config.selectors]
    assert list(result.selections) == names and len(result.overlap) == 6
    # Same inputs, same steps: the modular selector equals the correlation filter here.
    assert result.overlap[f"{names[0]}|{names[1]}"] == 1.0
    for name in names:
        boot = result.bootstrap[name]
        assert boot.n_failed == 0 and 0 < boot.stability <= 1
        assert boot.frequency["f1"] > 0.8
        assert boot.frequency["f9"] < boot.frequency["f1"]

    write_study(result, config, tmp_path)
    report = json.loads((tmp_path / "selection.json").read_text())
    assert report["n_participants"] == 300
    assert report["n_events"] + report["n_censored"] == 300
    assert report["selectors"][names[2]]["method"] == "lasso_cox"
    table = pd.read_csv(tmp_path / "selected_features.csv")
    assert set(table.columns) == {"selector", "rank", "feature", "score", "bootstrap_frequency"}
    assert table.groupby("selector")["rank"].min().eq(1).all()


def test_cli_runs_on_an_extracted_table(tmp_path, capsys):
    """Mechanics only: phantom features have no relation to the random follow-up."""
    manifest = phantoms.write_site(tmp_path / "site", 12, seed=4, rois={"box": 1},
                                   vary_masks=True)
    run_extraction(manifest, phantoms.config(), tmp_path / "out")
    raw = _config(n_features=4, min_events=3, reliability=None,
                  bootstrap={"n_resamples": 3, "seed": 1}).model_dump(mode="json")
    raw["selectors"] = raw["selectors"][:2]
    config_path = tmp_path / "selection.yaml"
    config_path.write_text(json.dumps(raw))  # JSON is valid YAML
    args = ["--features", str(tmp_path / "out/features.csv"), "--manifest", str(manifest),
            "--data-origin", "synthetic", "--config", str(config_path),
            "--output-dir", str(tmp_path / "selection"), "--log-level", "ERROR"]
    assert selection_cli(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["data_origin"] == "synthetic" and summary["n_participants"] == 12
    assert all(0 < n <= 4 for n in summary["selected"].values())
    assert (tmp_path / "selection" / "selected_features.csv").exists()

    # The shipped config filters on reliability: without estimates the CLI refuses.
    args[args.index("--config") + 1] = str(SHIPPED)
    assert selection_cli(args) == 2
    assert "no reliability estimates were provided" in capsys.readouterr().err
