"""Tests for the supervised dataset builder, its audit and the leakage guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oasis_radiomics.clinical.labels import (
    FUTURE_INFORMATION_COLUMNS,
    LABEL_AD,
    LABEL_CN,
    LABEL_MCI,
    LabelPolicy,
)
from oasis_radiomics.supervised_dataset import (
    SupervisedDatasetError,
    assert_no_subject_leakage,
    build_audit,
    build_diagnosis_vocabulary,
    build_progression_labels,
    build_session_labels,
    build_supervised_datasets,
    predictor_columns,
    subject_groups,
)


@pytest.fixture(scope="module")
def policy() -> LabelPolicy:
    """The shipped v2.0 policy: D1 primary, B4 auxiliary."""
    return LabelPolicy.load(None)


#: D1 flag sets standing in for the diagnoses the fixtures used to express in B4.
D1_BY_DIAGNOSIS = {
    "Cognitively normal": {"NORMCOG": "1"},
    "AD Dementia": {"DEMENTED": "1", "PROBAD": "1"},
    "Vascular Demt, primary": {"DEMENTED": "1", "VASC": "1"},
    "uncertain dementia": {"MCIAMEM": "1"},
    "a diagnosis nobody enumerated": {},
}


@pytest.fixture(scope="module")
def mci_policy() -> LabelPolicy:
    return LabelPolicy.from_mapping(
        {
            "version": "test-mci-v1",
            "current_state": {
                "CN": {"rule_id": "cn", "dx1_exact": ["Cognitively normal"]},
                "MCI": {"rule_id": "mci", "dx1_exact": ["uncertain dementia"]},
                "AD": {"rule_id": "ad", "dx1_exact": ["AD Dementia"]},
            },
            "excluded": {"OTHER_DEMENTIA": {"dx1_exact": ["Vascular Demt, primary"]}},
        }
    )


def _session(subject: str, day: int, dx1: str, *, valid: bool = True, gap: int = 10) -> dict:
    """One clinical-radiomics row carrying both the D1 flags and the B4 text."""
    return {
        **D1_BY_DIAGNOSIS.get(dx1, {}),
        "subject_id": subject,
        "session_id": f"{subject}_MR_d{day:04d}",
        "mri_day": day,
        "clinical_day": day + gap,
        "clinical_mri_abs_gap_days": abs(gap),
        "clinical_match_valid": valid,
        "dx1": dx1,
        "CDRTOT": "0",
        "age_at_mri": 70.0,
        "sex": "F",
        "original_glcm_Contrast_left_hippocampus": 1.5,
    }


# --- Target A over a table -------------------------------------------------
def test_every_input_row_produces_one_output_row(policy: LabelPolicy) -> None:
    rows = [
        _session("OAS30001", 100, "Cognitively normal"),
        _session("OAS30001", 900, "AD Dementia"),
        _session("OAS30002", 50, "Vascular Demt, primary"),
    ]
    labelled = build_session_labels(rows, policy)
    assert len(labelled) == 3
    assert [row["supervised_label"] for row in labelled] == [
        LABEL_CN,
        LABEL_AD,
        "OTHER_DEMENTIA",
    ]


def test_excluded_rows_are_kept_with_a_reason(policy: LabelPolicy) -> None:
    labelled = build_session_labels(
        [_session("OAS30001", 100, "Cognitively normal", valid=False)], policy
    )
    assert labelled[0]["training_eligible"] is False
    assert labelled[0]["training_exclusion_reason"] == "outside_clinical_window"


def test_raw_diagnosis_survives_next_to_the_label(policy: LabelPolicy) -> None:
    """Both the raw D1 flags and the raw B4 text survive the derivation."""
    labelled = build_session_labels([_session("OAS30001", 100, "AD Dementia")], policy)
    assert labelled[0]["dx1"] == "AD Dementia"
    assert labelled[0]["dx1_normalized"] == "ad dementia"
    assert labelled[0]["DEMENTED"] == "1"
    assert labelled[0]["PROBAD"] == "1"
    assert labelled[0]["supervised_label"] == LABEL_AD
    assert labelled[0]["b4_agreement"] == "agree"


def test_radiomic_features_are_preserved(policy: LabelPolicy) -> None:
    labelled = build_session_labels([_session("OAS30001", 100, "Cognitively normal")], policy)
    assert labelled[0]["original_glcm_Contrast_left_hippocampus"] == 1.5


def test_empty_input_is_rejected(policy: LabelPolicy) -> None:
    with pytest.raises(SupervisedDatasetError):
        build_session_labels([], policy)


# --- predictor columns and leakage ----------------------------------------
def test_predictors_exclude_labels_metadata_and_future(policy: LabelPolicy) -> None:
    labelled = build_session_labels([_session("OAS30001", 100, "Cognitively normal")], policy)
    predictors = predictor_columns(labelled[0].keys())

    assert "original_glcm_Contrast_left_hippocampus" in predictors
    assert "age_at_mri" in predictors
    for forbidden in ("supervised_label", "subject_id", "session_id", "dx1", "clinical_day"):
        assert forbidden not in predictors


def test_no_future_column_can_reach_the_predictors() -> None:
    """The central Target-B guarantee, asserted on the column contract itself."""
    columns = list(FUTURE_INFORMATION_COLUMNS) + ["original_firstorder_Mean_left"]
    assert predictor_columns(columns) == ["original_firstorder_Mean_left"]


def test_progression_rows_carry_no_future_predictors(mci_policy: LabelPolicy) -> None:
    sessions = build_session_labels([_session("S", 1000, "uncertain dementia")], mci_policy)
    visits = [
        {"subject_id": "S", "clinical_day": 1500, "dx1": "uncertain dementia"},
        {"subject_id": "S", "clinical_day": 2500, "dx1": "uncertain dementia"},
    ]
    rows = build_progression_labels(sessions, visits, mci_policy, horizon_days=1095)
    assert len(rows) == 1
    assert predictor_columns(rows[0].keys()) and not [
        column for column in predictor_columns(rows[0].keys())
        if column in FUTURE_INFORMATION_COLUMNS
    ]


# --- Target B over a table -------------------------------------------------
def test_only_mci_sessions_enter_the_progression_target(mci_policy: LabelPolicy) -> None:
    sessions = build_session_labels(
        [
            _session("S", 1000, "uncertain dementia"),
            _session("S", 1200, "Cognitively normal"),
            _session("T", 100, "AD Dementia"),
        ],
        mci_policy,
    )
    visits = [{"subject_id": "S", "clinical_day": 2500, "dx1": "uncertain dementia"}]
    rows = build_progression_labels(sessions, visits, mci_policy, horizon_days=1095)
    assert [row["session_id"] for row in rows] == ["S_MR_d1000"]


def test_progression_uses_only_the_same_subjects_visits(mci_policy: LabelPolicy) -> None:
    sessions = build_session_labels([_session("S", 1000, "uncertain dementia")], mci_policy)
    visits = [{"subject_id": "OTHER", "clinical_day": 1200, "dx1": "AD Dementia"}]
    rows = build_progression_labels(sessions, visits, mci_policy, horizon_days=1095)
    assert rows[0]["progression_label"] == "CENSORED"
    assert rows[0]["conversion_event"] is None


def test_shipped_policy_now_yields_progression_candidates(policy: LabelPolicy) -> None:
    """v2.0 resolves v1.0's blocker: D1 supplies an MCI class, so Target B runs."""
    sessions = build_session_labels([_session("S", 1000, "uncertain dementia")], policy)
    assert sessions[0]["supervised_label"] == LABEL_MCI

    visits = [{"subject_id": "S", "clinical_day": 2500, "MCIAMEM": "1"}]
    rows = build_progression_labels(sessions, visits, policy, 1095)
    assert len(rows) == 1
    assert rows[0]["progression_label"] == "MCI_STABLE"


# --- vocabulary audit ------------------------------------------------------
def test_vocabulary_covers_every_observed_string(policy: LabelPolicy) -> None:
    rows = [
        _session("OAS30001", 100, "Cognitively normal"),
        _session("OAS30002", 100, "AD Dementia"),
    ]
    rows[0]["dx2"] = "Active Mood disorder"
    vocabulary = build_diagnosis_vocabulary(rows, policy)
    observed = {(item["source_column"], item["raw_diagnosis"]) for item in vocabulary}
    assert ("dx1", "Cognitively normal") in observed
    assert ("dx1", "AD Dementia") in observed
    assert ("dx2", "Active Mood disorder") in observed


def test_vocabulary_flags_unmapped_strings(policy: LabelPolicy) -> None:
    vocabulary = build_diagnosis_vocabulary(
        [_session("OAS30001", 100, "a diagnosis nobody enumerated")], policy
    )
    entry = next(item for item in vocabulary if item["source_column"] == "dx1")
    assert entry["mapping_status"] == "UNMAPPED"
    assert entry["mapped_label"] == "UNMAPPED"


def test_vocabulary_counts_sessions_and_subjects(policy: LabelPolicy) -> None:
    rows = [
        _session("OAS30001", 100, "Cognitively normal"),
        _session("OAS30001", 900, "Cognitively normal"),
        _session("OAS30002", 100, "Cognitively normal"),
    ]
    entry = next(
        item
        for item in build_diagnosis_vocabulary(rows, policy)
        if item["raw_diagnosis"] == "Cognitively normal"
    )
    assert entry["count_sessions"] == 3
    assert entry["count_subjects"] == 2


# --- subject-level grouping -----------------------------------------------
def test_subject_groups_mirror_the_rows(policy: LabelPolicy) -> None:
    rows = build_session_labels(
        [_session("OAS30001", 100, "Cognitively normal"), _session("OAS30002", 50, "AD Dementia")],
        policy,
    )
    assert subject_groups(rows) == ["OAS30001", "OAS30002"]


def test_repeated_sessions_share_one_group(policy: LabelPolicy) -> None:
    """The whole point: two scans of one participant must not split."""
    rows = build_session_labels(
        [_session("OAS30001", 100, "Cognitively normal"), _session("OAS30001", 900, "AD Dementia")],
        policy,
    )
    assert len(set(subject_groups(rows))) == 1


def test_subject_leakage_is_detected() -> None:
    with pytest.raises(SupervisedDatasetError, match="Subject leakage"):
        assert_no_subject_leakage(["OAS30001", "OAS30002"], ["OAS30002"])


def test_disjoint_split_is_accepted() -> None:
    assert_no_subject_leakage(["OAS30001"], ["OAS30002"]) is None


# --- audit -----------------------------------------------------------------
def test_audit_counts_sessions_and_subjects(policy: LabelPolicy) -> None:
    rows = build_session_labels(
        [
            _session("OAS30001", 100, "Cognitively normal"),
            _session("OAS30001", 900, "Cognitively normal"),
            _session("OAS30002", 50, "AD Dementia"),
        ],
        policy,
    )
    audit = build_audit(rows, [], build_diagnosis_vocabulary(rows, policy), {})
    assert audit["sessions_total"] == 3
    assert audit["subjects_total"] == 2
    assert audit["CN_sessions"] == 2
    assert audit["CN_subjects"] == 1
    assert audit["AD_sessions"] == 1
    assert audit["training_eligible_subjects"] == 2


def test_audit_records_the_b4_cross_check(policy: LabelPolicy) -> None:
    rows = build_session_labels([_session("OAS30001", 100, "Cognitively normal")], policy)
    audit = build_audit(rows, [], build_diagnosis_vocabulary(rows, policy), {})
    assert audit["b4_agreement"]["agree"] == 1


def test_audit_marks_dataset_not_final_without_mci(policy: LabelPolicy) -> None:
    rows = build_session_labels([_session("OAS30001", 100, "Cognitively normal")], policy)
    audit = build_audit(rows, [], build_diagnosis_vocabulary(rows, policy), {})
    assert audit["dataset_is_final"] is False


# --- end to end ------------------------------------------------------------
def test_build_supervised_datasets_writes_all_four(tmp_path: Path, policy: LabelPolicy) -> None:
    sessions_csv = tmp_path / "sessions.csv"
    visits_csv = tmp_path / "visits.csv"
    sessions_csv.write_text(
        "subject_id,session_id,mri_day,clinical_day,clinical_mri_abs_gap_days,"
        "clinical_match_valid,NORMCOG,DEMENTED,PROBAD,dx1,CDRTOT,"
        "original_glcm_Contrast_left\n"
        "OAS30001,OAS30001_MR_d0100,100,110,10,True,1,,,Cognitively normal,0,1.5\n"
        "OAS30002,OAS30002_MR_d0050,50,60,10,True,,1,1,AD Dementia,1,1.7\n",
        encoding="utf-8",
    )
    visits_csv.write_text(
        "subject_id,clinical_day,NORMCOG,dx1\nOAS30001,110,1,Cognitively normal\n",
        encoding="utf-8",
    )

    result, outputs = build_supervised_datasets(sessions_csv, visits_csv, tmp_path / "out", policy)

    for name in (
        "supervised_radiomics_sessions",
        "supervised_mci_progression",
        "diagnosis_vocabulary",
        "supervised_label_audit",
    ):
        assert outputs[name].exists(), name

    audit = json.loads(outputs["supervised_label_audit"].read_text())
    assert audit["sessions_total"] == 2
    assert audit["CN_sessions"] == 1
    assert audit["AD_sessions"] == 1
    assert audit["parameters"]["label_policy_version"] == policy.version
    assert audit["parameters"]["progression_horizon_days"] == 1095
    assert result.session_rows[0]["supervised_label"] == LABEL_CN
