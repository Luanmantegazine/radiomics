"""Tests for the clinical imaging master table and the radiomics joins."""

from __future__ import annotations

from pathlib import Path

import pytest

from oasis_radiomics.clinical.models import ClinicalClassification, ClinicalVisit
from oasis_radiomics.clinical.trajectories import build_trajectory
from oasis_radiomics.clinical_dataset import (
    JOIN_KEYS,
    build_clinical_linkage,
    join_deltas,
    join_sessions,
    join_subjects,
    write_linkage_outputs,
)

CATALOGUE_CSV = """Label,Project,Date,Subject,M/F,Age,Type,Scanner,Scans
OAS30001_MR_d0100,OASIS3,,OAS30001,F,65,,3.0T,"T1w(1)"
OAS30001_MR_d1000,OASIS3,,OAS30001,F,68,,3.0T,"T1w(1)"
OAS30002_MR_d0050,OASIS3,,OAS30002,M,70,,3.0T,"T1w(1)"
"""

D1_CSV = """OASISID,OASIS_session_label,days_to_visit,age at visit,NORMCOG,PROBAD
OAS30001,OAS30001_UDSd1_d0120,0120,65.2,1,
OAS30001,OAS30001_UDSd1_d1010,1010,68.1,,1
"""

B4_CSV = """OASISID,OASIS_session_label,days_to_visit,age at visit,MMSE,CDRSUM,CDRTOT,dx1
OAS30001,OAS30001_UDSb4_d0120,120,65.2,29,0,0,Cognitively normal
OAS30001,OAS30001_UDSb4_d1010,1010,68.1,22,3,1,AD Dementia
"""


@pytest.fixture
def clinical_files(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name, content in (
        ("catalogue", CATALOGUE_CSV),
        ("d1", D1_CSV),
        ("b4", B4_CSV),
    ):
        path = tmp_path / f"{name}.csv"
        path.write_text(content, encoding="utf-8")
        paths[name] = path
    return paths


@pytest.fixture
def linkage(clinical_files: dict[str, Path]):
    return build_clinical_linkage(
        mri_catalog=clinical_files["catalogue"],
        d1_path=clinical_files["d1"],
        b4_path=clinical_files["b4"],
        window_days=180,
    )


# --- master table ----------------------------------------------------------
def test_master_has_one_row_per_mri_session(linkage) -> None:
    assert len(linkage.master_rows) == 3
    keys = [(row["subject_id"], row["session_id"]) for row in linkage.master_rows]
    assert len(set(keys)) == len(keys)


def test_master_keeps_sessions_without_clinical_data(linkage) -> None:
    """OAS30002 has no clinical visit at all and must still appear."""
    row = next(r for r in linkage.master_rows if r["subject_id"] == "OAS30002")
    assert row["clinical_match_found"] is False
    assert row["clinical_match_reason"] == "no_clinical_visit"
    assert row["clinical_match_valid"] is False


def test_master_carries_demographics(linkage) -> None:
    row = linkage.master_rows[0]
    assert row["sex"] == "F"
    assert row["scanner"] == "3.0T"
    assert row["age_at_mri"] == pytest.approx(65.0)


def test_master_preserves_raw_d1_and_b4_variables(linkage) -> None:
    """Derived columns never replace the raw diagnostic variables."""
    row = next(r for r in linkage.master_rows if r["session_id"] == "OAS30001_MR_d0100")
    assert row["NORMCOG"] == "1"
    assert row["MMSE"] == "29"
    assert row["CDRTOT"] == "0"
    assert row["dx1"] == "Cognitively normal"


def test_master_records_the_gap_with_its_sign(linkage) -> None:
    row = next(r for r in linkage.master_rows if r["session_id"] == "OAS30001_MR_d0100")
    assert row["clinical_day"] == 120
    assert row["clinical_mri_gap_days"] == 20
    assert row["clinical_mri_abs_gap_days"] == 20


def test_master_classification_is_unresolved_without_a_codebook(linkage) -> None:
    assert all(
        row["classification_status"] in ("unresolved_codebook", "no_clinical_data")
        for row in linkage.master_rows
    )
    assert all(row["cognitive_status"] == "UNKNOWN" for row in linkage.master_rows)


def test_session_without_clinical_data_states_the_reason(linkage) -> None:
    """An empty classification column would be indistinguishable from a bug."""
    row = next(r for r in linkage.master_rows if r["subject_id"] == "OAS30002")
    assert row["classification_status"] == "no_clinical_data"
    assert row["cognitive_status"] == "UNKNOWN"
    assert row["diagnosis_at_mri"] == "UNKNOWN"


def test_linkage_parameters_are_recorded_for_reproducibility(linkage) -> None:
    parameters = linkage.parameters
    assert parameters["clinical_window_days"] == 180
    assert parameters["matching_strategy"]
    assert parameters["clinical_linkage_version"]
    assert parameters["classification_frozen"] is False


def test_linkage_summary_counts(linkage) -> None:
    summary = linkage.summary
    assert summary["mri_sessions"] == 3
    assert summary["subjects"] == 2
    assert summary["valid_matches"] == 2
    assert summary["without_clinical_data"] == 1


def test_write_linkage_outputs(tmp_path: Path, linkage) -> None:
    outputs = write_linkage_outputs(linkage, tmp_path / "out")
    for name in ("clinical_visits", "clinical_imaging_master", "validation"):
        assert outputs[name].exists()

    import json

    report = json.loads(outputs["validation"].read_text())
    assert report["summary"]["mri_sessions"] == 3
    assert "parameters" in report


# --- radiomics joins -------------------------------------------------------
def _wide_rows():
    return [
        {
            "subject_id": "OAS30001",
            "session_id": "OAS30001_MR_d0100",
            "days_from_reference": 100,
            "original_glcm_Contrast_left_hippocampus": 1.5,
        },
        {
            "subject_id": "OAS30001",
            "session_id": "OAS30001_MR_d1000",
            "days_from_reference": 1000,
            "original_glcm_Contrast_left_hippocampus": 1.9,
        },
    ]


def test_join_sessions_is_one_row_per_subject_session(linkage) -> None:
    rows, issues = join_sessions(linkage.master_rows, _wide_rows())
    keys = [tuple(row[key] for key in JOIN_KEYS) for row in rows]
    assert len(rows) == 2
    assert len(set(keys)) == 2
    assert not [issue for issue in issues if issue.code == "duplicate_master_row"]


def test_join_sessions_keeps_clinical_and_radiomic_columns(linkage) -> None:
    rows, _ = join_sessions(linkage.master_rows, _wide_rows())
    row = rows[0]
    assert row["sex"] == "F"
    assert row["MMSE"] == "29"
    assert row["original_glcm_Contrast_left_hippocampus"] == 1.5
    assert row["clinical_master_found"] is True


def test_join_sessions_keeps_radiomics_without_clinical_master(linkage) -> None:
    """A radiomics session absent from the master is flagged, never dropped."""
    orphan = {
        "subject_id": "OAS39999",
        "session_id": "OAS39999_MR_d0001",
        "days_from_reference": 1,
        "original_glcm_Contrast_left_hippocampus": 2.0,
    }
    rows, issues = join_sessions(linkage.master_rows, _wide_rows() + [orphan])
    assert len(rows) == 3
    assert rows[-1]["clinical_master_found"] is False
    assert any(issue.code == "no_clinical_visit" for issue in issues)


def test_join_deltas_annotates_both_timepoints(linkage) -> None:
    visits = [ClinicalVisit("OAS30001", 120), ClinicalVisit("OAS30001", 1010)]
    classifications = {
        120: ClinicalClassification("CN", "NON_AD", "classified"),
        1010: ClinicalClassification("DEMENTIA", "AD", "classified"),
    }
    trajectory, _ = build_trajectory("OAS30001", visits, classifications)

    deltas = [
        {
            "subject_id": "OAS30001",
            "comparison": "baseline",
            "session_id_t0": "OAS30001_MR_d0100",
            "session_id_t1": "OAS30001_MR_d1000",
            "days_t0": 100,
            "days_t1": 1000,
            "delta_days": 900,
            "delta_years": 2.46,
            "delta_original_glcm_Contrast_left_hippocampus": 0.4,
            "slope_original_glcm_Contrast_left_hippocampus": 0.16,
        }
    ]
    row = join_deltas(linkage.master_rows, deltas, {"OAS30001": trajectory})[0]

    assert row["cdr_t0"] == "0"
    assert row["cdr_t1"] == "1"
    assert row["mmse_t0"] == "29"
    # The conversion is recorded at clinical day 1010, i.e. at the visit the
    # t1 scan (MRI day 1000) was matched to. It must still be detected.
    assert row["conversion_between_visits"] == "CN_to_AD"
    assert row["delta_original_glcm_Contrast_left_hippocampus"] == 0.4
    assert row["slope_original_glcm_Contrast_left_hippocampus"] == 0.16


def test_join_deltas_reports_no_conversion_outside_the_interval(linkage) -> None:
    visits = [ClinicalVisit("OAS30001", 120), ClinicalVisit("OAS30001", 1010)]
    classifications = {
        120: ClinicalClassification("CN", "NON_AD", "classified"),
        1010: ClinicalClassification("DEMENTIA", "AD", "classified"),
    }
    trajectory, _ = build_trajectory("OAS30001", visits, classifications)
    # Both endpoints sit after the conversion, so nothing happened in between.
    deltas = [
        {
            "subject_id": "OAS30001",
            "session_id_t0": "OAS30001_MR_d1000",
            "session_id_t1": "OAS30001_MR_d1000",
            "days_t0": 1010,
            "days_t1": 1500,
        }
    ]
    assert join_deltas(linkage.master_rows, deltas, {"OAS30001": trajectory})[0][
        "conversion_between_visits"
    ] is None


def test_join_subjects_attaches_the_trajectory(linkage) -> None:
    visits = [ClinicalVisit("OAS30001", 120), ClinicalVisit("OAS30001", 1010)]
    classifications = {
        120: ClinicalClassification("CN", "NON_AD", "classified"),
        1010: ClinicalClassification("DEMENTIA", "AD", "classified"),
    }
    trajectory, _ = build_trajectory("OAS30001", visits, classifications)

    slopes = [
        {
            "subject_id": "OAS30001",
            "feature": "original_glcm_Contrast_left_hippocampus",
            "feature_slope": 0.16,
            "feature_r2": 1.0,
            "n_sessions": 2,
        }
    ]
    row = join_subjects(
        linkage.master_rows, slopes, {"OAS30001": trajectory}, {"OAS30001": 2}
    )[0]

    assert row["baseline_diagnosis"] == "CN"
    assert row["last_diagnosis"] == "AD"
    assert row["clinical_trajectory"] == "CN -> AD"
    assert row["conversion_event"] == "CN_to_AD"
    assert row["n_mri_sessions"] == 2
    assert row["n_clinical_visits"] == 2
    assert row["feature_slope"] == 0.16
