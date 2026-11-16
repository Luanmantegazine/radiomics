"""Reliability: ICC known answers, repeated measures and the perturbation study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from client.reliability import IccForm, ReliabilityUnavailable, icc
from client.reliability.__main__ import main as reliability_cli
from client.reliability.repeated import assemble
from client.reliability.study import (
    ReliabilityStudyConfig,
    load_reliability,
    load_study_config,
    run_reliability_study,
)
from tests.federated import phantoms

# Shrout & Fleiss (1979), Table 2: 6 targets rated by 4 judges.
SHROUT_FLEISS = np.array(
    [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]],
    dtype=float,
)


@pytest.mark.parametrize(
    ("form", "estimate", "low", "high"),
    [
        (IccForm.ICC1, 0.166, -0.133, 0.723),
        (IccForm.ICC2, 0.290, 0.019, 0.761),
        (IccForm.ICC3, 0.715, 0.342, 0.946),
    ],
)
def test_icc_reproduces_shrout_and_fleiss(form, estimate, low, high):
    result = icc(SHROUT_FLEISS, form)
    assert result.estimate[0] == pytest.approx(estimate, abs=1e-3)
    assert result.ci_low[0] == pytest.approx(low, abs=2e-3)
    assert result.ci_high[0] == pytest.approx(high, abs=2e-3)
    assert (result.n_subjects, result.n_conditions) == (6, 4)


def test_icc_forms_differ_on_systematic_offsets():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=40)
    shifted = np.stack([truth, truth + 3.0], axis=1)  # second condition reads 3 units higher
    assert icc(shifted, IccForm.ICC3).estimate[0] == pytest.approx(1.0)
    assert icc(shifted, IccForm.ICC2).estimate[0] < 0.5
    identical = np.stack([truth, truth], axis=1)
    for form in IccForm:
        assert icc(identical, form).estimate[0] == pytest.approx(1.0)


def test_icc_is_vectorised_over_features_and_tracks_noise():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(200, 1, 3))
    noise = rng.normal(size=(200, 3, 3)) * np.array([0.1, 1.0, 5.0])
    result = icc(truth + noise, IccForm.ICC2)
    assert result.estimate.shape == (3,)
    assert result.estimate[0] > 0.95 > result.estimate[1] > 0.3 > result.estimate[2]
    # Theoretical ICC = 1 / (1 + noise variance).
    assert result.estimate[1] == pytest.approx(0.5, abs=0.08)
    assert (result.ci_low <= result.estimate).all() and (result.estimate <= result.ci_high).all()


def test_constant_feature_has_undefined_icc():
    assert np.isnan(icc(np.full((10, 2), 4.2), IccForm.ICC2).estimate[0])


def test_reliability_is_never_estimated_without_repeated_observations():
    single = np.random.default_rng(0).normal(size=(30, 1, 5))
    with pytest.raises(ReliabilityUnavailable, match="at least two observations"):
        icc(single, IccForm.ICC2)
    with pytest.raises(ReliabilityUnavailable, match="only 4 subjects"):
        icc(np.random.default_rng(0).normal(size=(4, 2, 5)), IccForm.ICC2, min_subjects=10)
    table = pd.DataFrame({"a": [1.0, 2.0]}, index=["P1", "P2"])
    with pytest.raises(ReliabilityUnavailable, match="at least two conditions"):
        assemble({"baseline": table})


def test_assemble_keeps_complete_cases_and_requires_one_schema():
    first = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=["P1", "P2", "P3"])
    second = pd.DataFrame({"a": [9.0, 1.5], "b": [9.0, 4.5]}, index=["P3", "P1"])
    repeated = assemble({"baseline": first, "rater2": second})
    assert repeated.participant_ids == ("P1", "P3")
    assert repeated.values.shape == (2, 2, 2)
    assert repeated.values[1].tolist() == [[3.0, 6.0], [9.0, 9.0]]
    with pytest.raises(ValueError, match="different feature schema"):
        assemble({"baseline": first, "other": second.rename(columns={"b": "c"})})
    with pytest.raises(ReliabilityUnavailable, match="no participant"):
        assemble({"baseline": first, "other": second.rename(index={"P1": "X", "P3": "Y"})})


# --------------------------------------------------------------------------- #
# Study on phantoms (mechanics only)
# --------------------------------------------------------------------------- #
def _study(**changes) -> ReliabilityStudyConfig:
    raw = load_study_config(
        phantoms.REPO / "experiments/configs/reliability/perturbation_v1.yaml"
    ).model_dump(mode="json")
    raw.update(changes)
    return ReliabilityStudyConfig.model_validate(raw)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("reliability")
    manifest = phantoms.write_site(root, 12, seed=5, rois={"box": 1}, vary_masks=True)
    summary = run_reliability_study(manifest, phantoms.config(), _study(), root / "out")
    return root, manifest, summary


def test_study_outputs(study):
    root, _, summary = study
    detail = pd.read_csv(root / "out" / "reliability.csv")
    assert set(detail["family"]) == {"segmentation_morphology", "discretisation"}
    assert len(detail) == 2 * 107 and len(summary) == 107
    assert (detail["n_subjects"] == 12).all() and (detail["n_conditions"] == 3).all()
    assert list(summary["feature"])[0] == "box__original_shape_Elongation"
    assert (root / "out/segmentation_morphology/erode_1/masks").is_dir()
    assert load_reliability(root / "out" / "reliability_summary.csv").keys() == set(
        summary["feature"]
    )


def test_each_family_measures_what_it_perturbs(study):
    root, _, _ = study
    detail = pd.read_csv(root / "out" / "reliability.csv").set_index(["family", "feature"])
    volume, mean = "box__original_shape_VoxelVolume", "box__original_firstorder_Mean"
    entropy = "box__original_firstorder_Entropy"
    # Changing the bin width cannot change shape or the mean, but changes entropy.
    assert detail.loc[("discretisation", volume), "icc"] == pytest.approx(1.0)
    assert detail.loc[("discretisation", mean), "icc"] == pytest.approx(1.0)
    assert detail.loc[("discretisation", entropy), "icc"] < 0.99
    # Eroding/dilating the mask systematically changes volume (absolute agreement).
    assert detail.loc[("segmentation_morphology", volume), "icc"] < 0.9


def test_summary_takes_the_worst_family(study):
    root, _, summary = study
    detail = pd.read_csv(root / "out" / "reliability.csv")
    expected = detail.groupby("feature")["icc"].min()
    got = summary.set_index("feature")["min_icc"]
    pd.testing.assert_series_equal(
        got.sort_index(), expected.sort_index(), check_names=False, rtol=1e-12
    )


def test_study_refuses_small_cohorts(tmp_path, capsys):
    manifest = phantoms.write_site(tmp_path, 4, seed=6, rois={"box": 1}, vary_masks=True)
    with pytest.raises(ReliabilityUnavailable):
        run_reliability_study(manifest, phantoms.config(), _study(), tmp_path / "out")
    code = reliability_cli(
        ["--manifest", str(manifest), "--extraction-config", str(phantoms.DEFAULT_CONFIG),
         "--study-config",
         str(phantoms.REPO / "experiments/configs/reliability/perturbation_v1.yaml"),
         "--output-dir", str(tmp_path / "cli"), "--log-level", "ERROR"]
    )
    assert code == 2 and "reliability not estimated" in capsys.readouterr().err
    assert not (tmp_path / "cli" / "reliability_summary.csv").exists()


def test_alternative_segmentations_are_supported(tmp_path):
    root = tmp_path / "site"
    manifest = phantoms.write_site(root, 10, seed=8, rois={"box": 1}, vary_masks=True)
    # A "second rater": the same participants with slightly different masks.
    rows = pd.read_csv(manifest, dtype=str).to_dict("records")
    for index, row in enumerate(rows):
        alternative = phantoms.labels(variant=index)
        alternative[4:6, 5:15, 6:10] = 0
        phantoms.write(alternative, root / "masks_r2" / f"{row['participant_id']}.nii.gz")
        row["mask_path"] = f"masks_r2/{row['participant_id']}.nii.gz"
    second = phantoms.write_manifest(root / "manifest_r2.csv", rows)
    config = _study(
        min_subjects=10,
        families=[{"name": "second_rater", "conditions": [
            {"kind": "alternative_manifest", "name": "rater2", "manifest_path": str(second)}]}],
    )
    summary = run_reliability_study(manifest, phantoms.config(), config, tmp_path / "out")
    values = summary.set_index("feature")["min_icc"]
    assert 0 < values["box__original_shape_VoxelVolume"] < 1
    assert summary["n_families"].eq(1).all()


def test_unknown_setting_override_is_rejected(tmp_path):
    manifest = phantoms.write_site(tmp_path, 10, seed=9, rois={"box": 1}, vary_masks=True)
    config = _study(families=[{"name": "typo", "conditions": [
        {"kind": "settings_override", "name": "x", "settings": {"binwidth": 20}}]}])
    with pytest.raises(ValueError, match="unknown PyRadiomics settings"):
        run_reliability_study(manifest, phantoms.config(), config, tmp_path / "out")
