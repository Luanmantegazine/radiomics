"""End-to-end test on the two locally available OASIS-3 pilot sessions.

Skipped automatically when the data is absent: OASIS-3 is credentialed and is
never committed to this repository.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from oasis_radiomics.config import PipelineConfig
from oasis_radiomics.longitudinal import HIPPOCAMPAL_VOLUME_TOTAL
from oasis_radiomics.pipeline import (
    DELTAS_CSV,
    LONG_CSV,
    QC_CSV,
    SLOPES_CSV,
    WIDE_CSV,
    run_pipeline,
)
from oasis_radiomics.tables import read_csv_rows

from .conftest import PILOT_SESSIONS, SMOKETEST_FREESURFER_DIR, requires_local_data

pytestmark = requires_local_data

EXPECTED_FEATURES_PER_ROI = 107


@pytest.fixture(scope="module")
def outputs(tmp_path_factory) -> dict[str, Path]:
    """Run the full pipeline once for every assertion in this module."""
    output_dir = tmp_path_factory.mktemp("pipeline_run")
    return run_pipeline(
        input_dir=SMOKETEST_FREESURFER_DIR,
        output_dir=output_dir,
        config=PipelineConfig.load(None),
    )


def test_all_expected_outputs_exist(outputs: dict[str, Path]) -> None:
    for name in (
        "features_long",
        "features_wide",
        "deltas",
        "slopes",
        "quality_control",
        "run_metadata",
    ):
        assert outputs[name].exists(), name

    assert outputs["features_long"].name == LONG_CSV
    assert outputs["features_wide"].name == WIDE_CSV
    assert outputs["deltas"].name == DELTAS_CSV
    assert outputs["slopes"].name == SLOPES_CSV
    assert outputs["quality_control"].name == QC_CSV


def test_long_table_has_two_rois_per_session(outputs: dict[str, Path]) -> None:
    rows = read_csv_rows(outputs["features_long"])
    assert len(rows) == 2 * len(PILOT_SESSIONS)
    assert {row["roi"] for row in rows} == {"left_hippocampus", "right_hippocampus"}
    assert {row["session_id"] for row in rows} == set(PILOT_SESSIONS)
    assert {row["subject_id"] for row in rows} == {"OAS30001"}


def test_no_bilateral_union_roi_is_extracted(outputs: dict[str, Path]) -> None:
    """The disconnected union mask must never reach PyRadiomics by default."""
    rows = read_csv_rows(outputs["features_long"])
    assert all("bilateral" not in str(row["roi"]) for row in rows)


def test_expected_feature_count(outputs: dict[str, Path]) -> None:
    row = read_csv_rows(outputs["features_long"])[0]
    features = [key for key in row if key.startswith("original_")]
    assert len(features) == EXPECTED_FEATURES_PER_ROI


def test_mask_voxel_counts_are_stable(outputs: dict[str, Path]) -> None:
    """Guards the aseg label mapping against silent regressions."""
    rows = {(row["session_id"], row["roi"]): row for row in read_csv_rows(outputs["features_long"])}
    assert rows[("OAS30001_MR_d0129", "left_hippocampus")]["mask_voxels"] == 3931
    assert rows[("OAS30001_MR_d0129", "right_hippocampus")]["mask_voxels"] == 4026
    assert rows[("OAS30001_MR_d0757", "left_hippocampus")]["mask_voxels"] == 3815
    assert rows[("OAS30001_MR_d0757", "right_hippocampus")]["mask_voxels"] == 4106


def test_wide_table_is_one_row_per_session_ordered_in_time(outputs: dict[str, Path]) -> None:
    rows = read_csv_rows(outputs["features_wide"])
    assert len(rows) == len(PILOT_SESSIONS)
    assert [row["days_from_reference"] for row in rows] == [129, 757]
    assert [row["session_id"] for row in rows] == list(PILOT_SESSIONS)


def test_wide_table_carries_the_derived_columns(outputs: dict[str, Path]) -> None:
    row = read_csv_rows(outputs["features_wide"])[0]
    for suffix in ("_left", "_right", "_mean", "_asymmetry"):
        assert f"original_glcm_Contrast{suffix}" in row
    assert "original_shape_MeshVolume_total" in row
    assert HIPPOCAMPAL_VOLUME_TOTAL in row
    assert row[HIPPOCAMPAL_VOLUME_TOTAL] == pytest.approx(3931 + 4026)


def test_wide_mean_and_asymmetry_are_consistent(outputs: dict[str, Path]) -> None:
    row = read_csv_rows(outputs["features_wide"])[0]
    left = row["original_glcm_Contrast_left"]
    right = row["original_glcm_Contrast_right"]
    assert row["original_glcm_Contrast_mean"] == pytest.approx((left + right) / 2)
    assert row["original_glcm_Contrast_asymmetry"] == pytest.approx(
        (left - right) / (left + right)
    )


def test_deltas_cover_the_two_visits(outputs: dict[str, Path]) -> None:
    rows = read_csv_rows(outputs["deltas"])
    assert {row["comparison"] for row in rows} == {"consecutive", "baseline"}
    row = rows[0]
    assert row["session_id_t0"] == "OAS30001_MR_d0129"
    assert row["session_id_t1"] == "OAS30001_MR_d0757"
    assert row["delta_days"] == 628
    assert row["delta_years"] == pytest.approx(628 / 365.25)


def test_delta_and_annual_rate_agree(outputs: dict[str, Path]) -> None:
    row = read_csv_rows(outputs["deltas"])[0]
    delta = row[f"delta_{HIPPOCAMPAL_VOLUME_TOTAL}"]
    rate = row[f"slope_{HIPPOCAMPAL_VOLUME_TOTAL}"]
    assert delta == pytest.approx((3815 + 4106) - (3931 + 4026))
    assert rate == pytest.approx(delta / (628 / 365.25))


def test_slopes_match_the_annualised_deltas(outputs: dict[str, Path]) -> None:
    """With two visits the OLS fit must reproduce delta / interval exactly."""
    deltas = {
        key: value
        for key, value in read_csv_rows(outputs["deltas"])[0].items()
        if key.startswith("slope_")
    }
    mismatches = 0
    for row in read_csv_rows(outputs["slopes"]):
        expected = deltas.get(f"slope_{row['feature']}")
        observed = row["feature_slope"]
        if expected is None:
            continue
        if math.isnan(observed) != math.isnan(expected):
            mismatches += 1
        elif not math.isnan(observed) and observed != pytest.approx(expected, rel=1e-9, abs=1e-12):
            mismatches += 1
    assert mismatches == 0


def test_slopes_report_two_sessions_and_perfect_fit(outputs: dict[str, Path]) -> None:
    row = next(
        item
        for item in read_csv_rows(outputs["slopes"])
        if item["feature"] == HIPPOCAMPAL_VOLUME_TOTAL
    )
    assert row["subject_id"] == "OAS30001"
    assert row["n_sessions"] == 2
    assert row["feature_r2"] == pytest.approx(1.0)
    assert row["followup_years"] == pytest.approx(628 / 365.25)


def test_quality_control_passes_for_both_sessions(outputs: dict[str, Path]) -> None:
    rows = read_csv_rows(outputs["quality_control"])
    assert len(rows) == len(PILOT_SESSIONS)
    assert all(row["qc_status"] == "pass" for row in rows)
    assert all(row["qc_outlier"] is False or row["qc_outlier"] == False for row in rows)
    assert all("insufficient_samples" in row["qc_outlier_reason"] for row in rows)


def test_run_metadata_records_the_environment(outputs: dict[str, Path]) -> None:
    metadata = json.loads(outputs["run_metadata"].read_text())
    assert metadata["counts"]["subjects"] == 1
    assert metadata["counts"]["sessions"] == 2
    assert metadata["counts"]["features_per_roi"] == EXPECTED_FEATURES_PER_ROI
    assert metadata["counts"]["failed_sessions"] == []
    assert metadata["environment"]["numpy"].startswith("1.26")
    assert "pyradiomics" in metadata["environment"]
    assert metadata["config"]["radiomics"]["binWidth"] == 25
    assert metadata["timestamp_utc"]
