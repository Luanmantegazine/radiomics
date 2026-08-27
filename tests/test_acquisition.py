"""Tests for acquisition manifest construction and final schema validation."""

from __future__ import annotations

from pathlib import Path

from oasis_radiomics.acquisition import (
    CatalogueSession,
    eligible_subjects,
    parse_freesurfer_id,
    select_subjects,
    validate_long_feature_rows,
)
from oasis_radiomics.protocol import ALZHEIMER_ROIS, expected_feature_keys
from prepare_acquisition import _read_input_catalogue


def test_parse_freesurfer_id() -> None:
    item = parse_freesurfer_id("OAS30001_Freesurfer53_d0129")
    assert item.subject_id == "OAS30001"
    assert item.days_from_reference == 129
    assert item.freesurfer_version == "53"


def test_official_mri_catalogue_is_converted_to_freesurfer53_candidates(tmp_path: Path) -> None:
    path = tmp_path / "oasis_mri.csv"
    path.write_text(
        "Label,Project,Date,Subject,M/F,Age,Type,Scanner,Scans\n"
        'OAS30001_MR_d0129,OASIS3,,OAS30001,F,65,,3.0T,"T1w(2), T2w(2)"\n'
        'OAS30001_MR_d0757,OASIS3,,OAS30001,F,67,,3.0T,"T1w(2), dwi(1)"\n'
        'OAS30002_MR_d0653,OASIS3,,OAS30002,M,68,,1.5T,"T1w(1)"\n'
        'OAS30003_MR_d0558,OASIS3,,OAS30003,F,60,,3.0T,"T2w(2), bold(3)"\n',
        encoding="utf-8",
    )

    sessions, source_format = _read_input_catalogue(path, "53")

    assert source_format == "mri_candidates"
    assert [item.freesurfer_id for item in sessions] == [
        "OAS30001_Freesurfer53_d0129",
        "OAS30001_Freesurfer53_d0757",
    ]


def test_longitudinal_eligibility_and_reproducible_selection() -> None:
    sessions = [
        CatalogueSession("OAS30001_Freesurfer53_d0001", "OAS30001", 1, "53"),
        CatalogueSession("OAS30001_Freesurfer53_d0100", "OAS30001", 100, "53"),
        CatalogueSession("OAS30002_Freesurfer53_d0002", "OAS30002", 2, "53"),
        CatalogueSession("OAS30003_Freesurfer53_d0003", "OAS30003", 3, "53"),
        CatalogueSession("OAS30003_Freesurfer53_d0200", "OAS30003", 200, "53"),
    ]
    eligible = eligible_subjects(sessions, min_sessions=2)
    assert set(eligible) == {"OAS30001", "OAS30003"}
    assert select_subjects(eligible, target_subjects=1, oversample=1.0, seed=2026) == select_subjects(
        eligible, target_subjects=1, oversample=1.0, seed=2026
    )


def _valid_session_rows() -> list[dict[str, object]]:
    features = {key: 1.0 for key in expected_feature_keys()}
    rows = []
    for roi in ALZHEIMER_ROIS:
        row: dict[str, object] = {
            "subject_id": "OAS30001",
            "session_id": "OAS30001_MR_d0129",
            "days_from_reference": 129,
            "roi": roi,
            "mask_voxels": 1000,
            "mask_volume_mm3": 1000.0,
        }
        row.update(features)
        rows.append(row)
    return rows


def test_validated_session_requires_16_by_107() -> None:
    summary = validate_long_feature_rows(_valid_session_rows())
    assert summary.sessions_seen == 1
    assert summary.valid_sessions == 1
    assert summary.invalid_sessions == 0
    assert summary.expected_raw_features_per_session == 1712


def test_missing_roi_fails_acquisition_gate() -> None:
    rows = _valid_session_rows()[:-1]
    summary = validate_long_feature_rows(rows)
    assert summary.invalid_sessions == 1
    assert any(issue.code == "missing_rois" for issue in summary.issues)


def test_missing_feature_fails_acquisition_gate() -> None:
    rows = _valid_session_rows()
    first_feature = expected_feature_keys()[0]
    rows[0].pop(first_feature)
    summary = validate_long_feature_rows(rows)
    assert summary.invalid_sessions == 1
    assert any(issue.code == "feature_schema" for issue in summary.issues)
