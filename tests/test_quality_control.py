"""Tests for per-session quality control and outlier flagging."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oasis_radiomics.config import OutlierConfig, QualityControlConfig
from oasis_radiomics.discovery import DiscoveredSession
from oasis_radiomics.ids import parse_session_id
from oasis_radiomics.masks import LoadedVolume, describe_mask
from oasis_radiomics.quality_control import (
    QC_FAIL,
    QC_PASS,
    QC_WARN,
    check_image_geometry,
    check_mask,
    check_session,
    flag_outliers,
    iqr_bounds,
    robust_z_scores,
)

from pathlib import Path

QC_CONFIG = QualityControlConfig()


def _volume(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0), fill=100.0) -> LoadedVolume:
    return LoadedVolume(
        data=np.full(shape, fill, dtype=np.float32),
        affine=np.eye(4),
        spacing=spacing,
        path=Path("in-memory"),
    )


def _hippocampus_mask(shape=(16, 16, 16), n_voxels=2000) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask.reshape(-1)[:n_voxels] = 1
    return mask


def _session(session_id: str = "OAS30001_MR_d0129") -> DiscoveredSession:
    return DiscoveredSession(
        key=parse_session_id(session_id),
        t1_path=Path(f"/tmp/{session_id}/mri/T1.mgz"),
        aseg_path=Path(f"/tmp/{session_id}/mri/aseg.mgz"),
    )


def _masks(left_voxels: int = 2000, right_voxels: int = 2050, shape=(16, 16, 16)) -> dict:
    return {
        "left_hippocampus": describe_mask(
            "left_hippocampus", [17], _hippocampus_mask(shape, left_voxels), 1.0
        ),
        "right_hippocampus": describe_mask(
            "right_hippocampus", [53], _hippocampus_mask(shape, right_voxels), 1.0
        ),
    }


# --- geometry --------------------------------------------------------------
def test_geometry_accepts_matching_volumes() -> None:
    errors, warnings = check_image_geometry(_volume(), _volume())
    assert errors == []
    assert warnings == []


def test_geometry_reports_shape_mismatch_as_error() -> None:
    errors, _ = check_image_geometry(_volume((16, 16, 16)), _volume((8, 8, 8)))
    assert any("shape_mismatch" in error for error in errors)


def test_geometry_reports_affine_mismatch_as_warning() -> None:
    aseg = _volume()
    shifted = LoadedVolume(aseg.data, np.eye(4) * 2, aseg.spacing, aseg.path)
    errors, warnings = check_image_geometry(_volume(), shifted)
    assert errors == []
    assert any("affine_mismatch" in warning for warning in warnings)


def test_geometry_rejects_all_zero_image() -> None:
    errors, _ = check_image_geometry(_volume(fill=0.0), _volume())
    assert "t1_all_zero" in errors


def test_geometry_rejects_invalid_spacing() -> None:
    errors, _ = check_image_geometry(_volume(spacing=(0.0, 1.0, 1.0)), _volume())
    assert any("invalid_voxel_spacing" in error for error in errors)


# --- masks -----------------------------------------------------------------
def test_check_mask_accepts_plausible_hippocampus() -> None:
    roi = describe_mask("left_hippocampus", [17], _hippocampus_mask(), 1.0)
    assert check_mask(roi, (16, 16, 16), QC_CONFIG) == []


def test_check_mask_reports_empty_mask() -> None:
    roi = describe_mask("left_hippocampus", [17], np.zeros((16, 16, 16), np.uint8), 1.0)
    assert check_mask(roi, (16, 16, 16), QC_CONFIG) == ["left_hippocampus_empty"]


def test_check_mask_reports_implausible_volumes() -> None:
    small = describe_mask("left_hippocampus", [17], _hippocampus_mask(n_voxels=100), 1.0)
    large = describe_mask(
        "right_hippocampus", [53], _hippocampus_mask((32, 32, 32), n_voxels=9000), 1.0
    )
    assert any("few_voxels" in problem for problem in check_mask(small, (16, 16, 16), QC_CONFIG))
    assert any("volume_below" in problem for problem in check_mask(small, (16, 16, 16), QC_CONFIG))
    assert any("volume_above" in problem for problem in check_mask(large, (32, 32, 32), QC_CONFIG))


def test_check_mask_reports_out_of_bounds() -> None:
    roi = describe_mask("left_hippocampus", [17], _hippocampus_mask((16, 16, 16)), 1.0)
    problems = check_mask(roi, (8, 8, 8), QC_CONFIG)
    assert any("shape_mismatch" in problem for problem in problems)


# --- session ---------------------------------------------------------------
def test_check_session_passes_on_healthy_input() -> None:
    qc = check_session(_session(), _volume(), _volume(), _masks(), QC_CONFIG)
    assert qc.status == QC_PASS
    assert qc.usable
    assert qc.left_voxels == 2000
    assert qc.right_voxels == 2050
    assert qc.total_volume_mm3 == 4050.0
    assert qc.volume_asymmetry == pytest.approx(-50.0 / 4050.0)
    assert qc.subject_id == "OAS30001"
    assert qc.days_from_reference == 129


def test_check_session_fails_on_empty_hippocampus() -> None:
    masks = _masks()
    masks["left_hippocampus"] = describe_mask(
        "left_hippocampus", [17], np.zeros((16, 16, 16), np.uint8), 1.0
    )
    qc = check_session(_session(), _volume(), _volume(), masks, QC_CONFIG)
    assert qc.status == QC_FAIL
    assert not qc.usable
    assert math.isnan(qc.volume_asymmetry)


def test_check_session_fails_on_missing_roi() -> None:
    qc = check_session(_session(), _volume(), _volume(), {}, QC_CONFIG)
    assert qc.status == QC_FAIL
    assert "left_hippocampus_missing" in qc.errors


def test_check_session_warns_on_extreme_asymmetry() -> None:
    qc = check_session(
        _session(), _volume(), _volume(), _masks(left_voxels=1200, right_voxels=2400), QC_CONFIG
    )
    assert qc.status == QC_WARN
    assert qc.usable  # a warning never blocks extraction
    assert any("high_volume_asymmetry" in warning for warning in qc.warnings)


def test_session_qc_row_has_the_expected_columns() -> None:
    row = check_session(_session(), _volume(), _volume(), _masks(), QC_CONFIG).as_row()
    for column in (
        "subject_id",
        "session_id",
        "left_voxels",
        "right_voxels",
        "left_volume_mm3",
        "right_volume_mm3",
        "qc_status",
        "qc_warning",
    ):
        assert column in row


# --- robust statistics -----------------------------------------------------
def test_robust_z_scores_flags_the_extreme_value() -> None:
    scores = robust_z_scores([10.0, 11.0, 10.5, 9.5, 100.0])
    assert abs(scores[-1]) > 3.5
    assert all(abs(score) < 3.5 for score in scores[:-1])


def test_robust_z_scores_returns_nan_when_mad_is_zero() -> None:
    assert np.all(np.isnan(robust_z_scores([5.0, 5.0, 5.0, 5.0])))


def test_robust_z_scores_handles_all_nan() -> None:
    assert np.all(np.isnan(robust_z_scores([float("nan")] * 3)))


def test_iqr_bounds() -> None:
    low, high = iqr_bounds([1.0, 2.0, 3.0, 4.0], 1.5)
    assert low < 1.0 < 4.0 < high


# --- cohort outlier flagging -----------------------------------------------
def _qc_rows(volumes: list[float]) -> list[dict]:
    return [
        {"session_id": f"OAS3000{index}_MR_d0001", "left_volume_mm3": volume}
        for index, volume in enumerate(volumes)
    ]


def test_flag_outliers_marks_extreme_session() -> None:
    rows = _qc_rows([3900, 3950, 4000, 3980, 3920, 3890, 4010, 20000])
    flagged = flag_outliers(rows, OutlierConfig(columns=("left_volume_mm3",), min_samples=8))
    assert [row["qc_outlier"] for row in flagged] == [False] * 7 + [True]
    assert "left_volume_mm3" in flagged[-1]["qc_outlier_reason"]


def test_flag_outliers_never_removes_rows() -> None:
    rows = _qc_rows([3900, 3950, 4000, 3980, 3920, 3890, 4010, 20000])
    assert len(flag_outliers(rows, OutlierConfig(min_samples=8, columns=("left_volume_mm3",)))) == 8


def test_flag_outliers_reports_insufficient_samples() -> None:
    flagged = flag_outliers(_qc_rows([3900, 4000]), OutlierConfig(min_samples=8))
    assert all(row["qc_outlier"] is False for row in flagged)
    assert all("insufficient_samples" in row["qc_outlier_reason"] for row in flagged)


def test_flag_outliers_can_be_disabled() -> None:
    flagged = flag_outliers(_qc_rows([3900, 4000]), OutlierConfig(method="none"))
    assert all(row["qc_outlier_reason"] == "disabled" for row in flagged)


def test_flag_outliers_iqr_method() -> None:
    rows = _qc_rows([3900, 3950, 4000, 3980, 3920, 3890, 4010, 20000])
    flagged = flag_outliers(
        rows, OutlierConfig(method="iqr", min_samples=8, columns=("left_volume_mm3",))
    )
    assert flagged[-1]["qc_outlier"] is True


def test_flag_outliers_ignores_missing_column() -> None:
    flagged = flag_outliers(
        _qc_rows([1.0] * 10), OutlierConfig(min_samples=8, columns=("does_not_exist",))
    )
    assert all(row["qc_outlier"] is False for row in flagged)
