"""Tests for the bilateral and longitudinal derivations."""

from __future__ import annotations

import math

import pytest

from oasis_radiomics.config import BilateralConfig, LongitudinalConfig
from oasis_radiomics.longitudinal import (
    HIPPOCAMPAL_VOLUME_TOTAL,
    LongitudinalError,
    build_wide_table,
    calculate_annualised_rate,
    calculate_asymmetry,
    calculate_delta,
    calculate_mean,
    calculate_slope,
    calculate_total,
    compute_deltas,
    compute_slopes,
)

NAN = float("nan")


# --- scalar helpers --------------------------------------------------------
def test_calculate_mean_and_total() -> None:
    assert calculate_mean(4000.0, 4200.0) == 4100.0
    assert calculate_total(4000.0, 4200.0) == 8200.0


def test_calculate_mean_propagates_missing() -> None:
    assert math.isnan(calculate_mean(NAN, 4200.0))
    assert math.isnan(calculate_total(4000.0, NAN))


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (4000.0, 4000.0, 0.0),
        (4200.0, 3800.0, 0.05),
        (3800.0, 4200.0, -0.05),
        (1.0, 3.0, -0.5),
    ],
)
def test_calculate_asymmetry(left: float, right: float, expected: float) -> None:
    assert calculate_asymmetry(left, right) == pytest.approx(expected)


def test_asymmetry_is_antisymmetric() -> None:
    assert calculate_asymmetry(4200.0, 3800.0) == pytest.approx(
        -calculate_asymmetry(3800.0, 4200.0)
    )


def test_asymmetry_handles_division_by_zero() -> None:
    assert math.isnan(calculate_asymmetry(0.0, 0.0))
    assert math.isnan(calculate_asymmetry(5.0, -5.0, mode="always"))


def test_asymmetry_positive_only_rejects_signed_values() -> None:
    """Signed texture features must not get a meaningless asymmetry index."""
    assert math.isnan(calculate_asymmetry(-0.21, -0.13))
    assert math.isnan(calculate_asymmetry(-0.21, 0.13))


def test_asymmetry_always_mode_flips_sign_on_negative_values() -> None:
    """Why 'positive_only' is the default: the index inverts below zero.

    ``left`` is smaller than ``right`` here, yet the normalised difference comes
    out positive, because the denominator is negative.
    """
    assert calculate_asymmetry(-0.3, -0.1, mode="always") == pytest.approx(+0.5)
    assert calculate_asymmetry(0.3, 0.1, mode="always") == pytest.approx(+0.5)
    assert math.isnan(calculate_asymmetry(-0.3, -0.1, mode="positive_only"))


def test_asymmetry_propagates_missing() -> None:
    assert math.isnan(calculate_asymmetry(NAN, 1.0))


def test_asymmetry_rejects_unknown_mode() -> None:
    with pytest.raises(LongitudinalError):
        calculate_asymmetry(1.0, 2.0, mode="whatever")


def test_calculate_delta() -> None:
    assert calculate_delta(4000.0, 3900.0) == -100.0
    assert calculate_delta(2.0, 5.0) == 3.0
    assert math.isnan(calculate_delta(NAN, 5.0))


def test_calculate_annualised_rate() -> None:
    assert calculate_annualised_rate(-100.0, 2.0) == -50.0
    assert math.isnan(calculate_annualised_rate(-100.0, 0.0))
    assert math.isnan(calculate_annualised_rate(NAN, 2.0))


# --- slope -----------------------------------------------------------------
def test_calculate_slope_perfect_line() -> None:
    fit = calculate_slope([0.0, 1.0, 2.0, 3.0], [10.0, 8.0, 6.0, 4.0])
    assert fit["slope"] == pytest.approx(-2.0)
    assert fit["intercept"] == pytest.approx(10.0)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["n_points"] == 4
    assert fit["time_span"] == pytest.approx(3.0)


def test_calculate_slope_two_points_matches_annualised_delta() -> None:
    """With two visits the fit must reduce to delta / interval."""
    times, values = [0.0, 1.72], [4000.0, 3900.0]
    fit = calculate_slope(times, values)
    expected = calculate_annualised_rate(calculate_delta(values[0], values[1]), 1.72)
    assert fit["slope"] == pytest.approx(expected)
    assert fit["r2"] == pytest.approx(1.0)


def test_calculate_slope_ignores_non_finite_pairs() -> None:
    fit = calculate_slope([0.0, 1.0, 2.0], [10.0, NAN, 6.0])
    assert fit["slope"] == pytest.approx(-2.0)
    assert fit["n_points"] == 2


def test_calculate_slope_needs_two_distinct_times() -> None:
    assert math.isnan(calculate_slope([1.0], [4.0])["slope"])
    assert math.isnan(calculate_slope([1.0, 1.0], [4.0, 9.0])["slope"])
    assert math.isnan(calculate_slope([0.0, 1.0], [NAN, NAN])["slope"])


def test_calculate_slope_rejects_mismatched_lengths() -> None:
    with pytest.raises(LongitudinalError):
        calculate_slope([0.0, 1.0], [1.0])


def test_calculate_slope_noisy_r2_below_one() -> None:
    fit = calculate_slope([0.0, 1.0, 2.0, 3.0], [10.0, 7.0, 7.0, 4.0])
    assert 0.0 < fit["r2"] < 1.0


# --- wide table ------------------------------------------------------------
def test_build_wide_table_shapes(long_rows: list[dict]) -> None:
    wide = build_wide_table(long_rows, BilateralConfig())
    assert len(wide) == 3
    assert [row["days_from_reference"] for row in wide] == [129, 757, 1400]
    assert wide[0]["subject_id"] == "OAS39999"


def test_build_wide_table_derived_columns(long_rows: list[dict]) -> None:
    row = build_wide_table(long_rows, BilateralConfig())[0]
    assert row["mask_volume_mm3_left"] == 4000.0
    assert row["mask_volume_mm3_right"] == 4200.0
    assert row["mask_volume_mm3_mean"] == 4100.0
    assert row["mask_volume_mm3_total"] == 8200.0
    assert row["mask_volume_mm3_asymmetry"] == pytest.approx(-200.0 / 8200.0)
    assert row[HIPPOCAMPAL_VOLUME_TOTAL] == 8200.0


def test_build_wide_table_only_additive_features_get_total(long_rows: list[dict]) -> None:
    row = build_wide_table(long_rows, BilateralConfig())[0]
    assert "original_shape_MeshVolume_total" in row
    assert "original_firstorder_Entropy_total" not in row
    assert "original_firstorder_Entropy_mean" in row


def test_build_wide_table_signed_feature_has_nan_asymmetry(long_rows: list[dict]) -> None:
    row = build_wide_table(long_rows, BilateralConfig())[0]
    assert math.isnan(row["original_firstorder_Skewness_asymmetry"])
    assert not math.isnan(row["original_firstorder_Skewness_mean"])


def test_build_wide_table_drops_roi_and_path_columns(long_rows: list[dict]) -> None:
    row = build_wide_table(long_rows, BilateralConfig())[0]
    assert "roi" not in row
    assert not any(column.startswith("image_path") for column in row)


def test_build_wide_table_missing_side_yields_nan(long_rows: list[dict]) -> None:
    only_left = [row for row in long_rows if row["roi"] == "left_hippocampus"]
    row = build_wide_table(only_left, BilateralConfig())[0]
    assert row["mask_volume_mm3_left"] == 4000.0
    assert math.isnan(row["mask_volume_mm3_right"])
    assert math.isnan(row["mask_volume_mm3_mean"])


def test_build_wide_table_keeps_extra_roi_without_deriving(long_rows: list[dict]) -> None:
    extra = dict(long_rows[0])
    extra["roi"] = "bilateral_hippocampus"
    row = build_wide_table(long_rows + [extra], BilateralConfig())[0]
    assert "mask_volume_mm3_bilateral_hippocampus" in row
    assert row["mask_volume_mm3_total"] == 8200.0  # unaffected by the union ROI


def test_build_wide_table_rejects_empty_input() -> None:
    with pytest.raises(LongitudinalError):
        build_wide_table([], BilateralConfig())


# --- deltas ----------------------------------------------------------------
def test_compute_deltas_consecutive_and_baseline(long_rows: list[dict]) -> None:
    wide = build_wide_table(long_rows, BilateralConfig())
    deltas = compute_deltas(wide, LongitudinalConfig())

    consecutive = [row for row in deltas if row["comparison"] == "consecutive"]
    baseline = [row for row in deltas if row["comparison"] == "baseline"]
    assert len(consecutive) == 2
    assert len(baseline) == 2
    assert [row["delta_days"] for row in consecutive] == [628, 643]
    assert [row["delta_days"] for row in baseline] == [628, 1271]


def test_compute_deltas_values(long_rows: list[dict]) -> None:
    wide = build_wide_table(long_rows, BilateralConfig())
    row = next(
        item
        for item in compute_deltas(wide, LongitudinalConfig())
        if item["comparison"] == "baseline" and item["delta_days"] == 1271
    )
    assert row[f"delta_{HIPPOCAMPAL_VOLUME_TOTAL}"] == pytest.approx(-400.0)
    assert row["delta_years"] == pytest.approx(1271 / 365.25)
    assert row[f"slope_{HIPPOCAMPAL_VOLUME_TOTAL}"] == pytest.approx(-400.0 / (1271 / 365.25))


def test_compute_deltas_skips_single_session_subjects(long_rows: list[dict]) -> None:
    single = [row for row in long_rows if row["days_from_reference"] == 129]
    wide = build_wide_table(single, BilateralConfig())
    assert compute_deltas(wide, LongitudinalConfig()) == []


def test_compute_deltas_same_day_gives_nan_rate() -> None:
    wide = [
        {"subject_id": "S", "session_id": "a", "days_from_reference": 10, "feature": 1.0},
        {"subject_id": "S", "session_id": "b", "days_from_reference": 10, "feature": 2.0},
    ]
    row = compute_deltas(wide, LongitudinalConfig(delta_comparisons=("consecutive",)))[0]
    assert row["delta_feature"] == 1.0
    assert math.isnan(row["slope_feature"])


def test_compute_deltas_unsorted_input_is_reordered(long_rows: list[dict]) -> None:
    wide = build_wide_table(long_rows, BilateralConfig())
    reversed_wide = list(reversed(wide))
    deltas = compute_deltas(reversed_wide, LongitudinalConfig(delta_comparisons=("baseline",)))
    assert all(row["days_t0"] == 129 for row in deltas)


# --- slopes ----------------------------------------------------------------
def test_compute_slopes_per_subject_and_feature(long_rows: list[dict]) -> None:
    wide = build_wide_table(long_rows, BilateralConfig())
    slopes = compute_slopes(wide, LongitudinalConfig())

    row = next(item for item in slopes if item["feature"] == HIPPOCAMPAL_VOLUME_TOTAL)
    assert row["subject_id"] == "OAS39999"
    assert row["n_sessions"] == 3
    assert row["followup_years"] == pytest.approx(1271 / 365.25)
    assert row["feature_slope"] < 0
    assert row["baseline_value"] == 8200.0
    assert row["last_value"] == 7800.0


def test_compute_slopes_respects_min_sessions(long_rows: list[dict]) -> None:
    wide = build_wide_table(long_rows, BilateralConfig())
    assert compute_slopes(wide, LongitudinalConfig(min_sessions_for_slope=4)) == []


def test_compute_slopes_all_nan_feature_reports_zero_sessions(long_rows: list[dict]) -> None:
    wide = build_wide_table(long_rows, BilateralConfig())
    row = next(
        item
        for item in compute_slopes(wide, LongitudinalConfig())
        if item["feature"] == "original_firstorder_Skewness_asymmetry"
    )
    assert row["n_sessions"] == 0
    assert math.isnan(row["feature_slope"])
