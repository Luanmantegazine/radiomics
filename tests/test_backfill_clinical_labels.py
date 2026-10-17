"""Tests for scripts/backfill_clinical_labels.py using synthetic rows only."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.backfill_clinical_labels import (
    CAT_A,
    CAT_B,
    CAT_C,
    CAT_D,
    CAT_E,
    MASTER_FLAG,
    NOTE_COLUMN,
    BackfillError,
    allowed_columns,
    assert_lossless_roundtrip,
    classify_unlabelled,
    column_style,
    compute_counts,
    mirror_style,
    plan_cells,
    read_text_table,
    serialise_table,
    verify_backfill,
)

WINDOW = 180


def fresh(**overrides: str) -> dict[str, str]:
    """A fresh production-path row for a labelled, validly paired session."""
    base = {
        "clinical_match_found": "True",
        "clinical_match_valid": "True",
        "clinical_match_reason": "matched",
        "clinical_mri_abs_gap_days": "40",
        "label_status": "labelled",
        "label_rule_id": "d1_normcog",
        "label_reason": "reason",
    }
    base.update(overrides)
    return base


# --- classification --------------------------------------------------------
def test_subject_without_any_clinical_file_is_category_a() -> None:
    category, _ = classify_unlabelled(
        {}, fresh(clinical_match_found="False", clinical_match_valid="False",
                  clinical_match_reason="no_clinical_visit"), False, WINDOW,
    )
    assert category == CAT_A


def test_subject_only_in_c1_is_category_e_not_a() -> None:
    category, detail = classify_unlabelled(
        {}, fresh(clinical_match_found="False", clinical_match_valid="False",
                  clinical_match_reason="no_clinical_visit"), True, WINDOW,
    )
    assert category == CAT_E
    assert "no D1/B4 visit" in detail


def test_nearest_visit_outside_window_is_category_b_and_reports_the_gap() -> None:
    category, detail = classify_unlabelled(
        {}, fresh(clinical_match_valid="False", clinical_match_reason="outside_window",
                  clinical_mri_abs_gap_days="196"), True, WINDOW,
    )
    assert category == CAT_B
    assert "196" in detail


def test_valid_pairing_absent_from_v2_is_category_c() -> None:
    category, _ = classify_unlabelled({"clinical_match_found": ""}, fresh(), True, WINDOW)
    assert category == CAT_C


def test_policy_that_does_not_decide_is_category_d_even_if_pairing_was_absent() -> None:
    row = fresh(label_status="missing_diagnosis", label_rule_id="d1_no_cognitive_status")
    category, detail = classify_unlabelled({"clinical_match_found": ""}, row, True, WINDOW)
    assert category == CAT_D
    assert "d1_no_cognitive_status" in detail


def test_pairing_already_in_v2_but_label_never_generated_is_category_e() -> None:
    category, detail = classify_unlabelled({"clinical_match_found": "True"}, fresh(), True, WINDOW)
    assert category == CAT_E
    assert "never ran" in detail


def test_session_missing_from_the_catalogue_run_is_category_e() -> None:
    category, _ = classify_unlabelled({}, None, True, WINDOW)
    assert category == CAT_E


# --- style mirroring -------------------------------------------------------
def test_integral_float_columns_are_detected() -> None:
    assert column_style(["129.0", "", "757.0"]) == "float0"
    assert column_style(["129", "757"]) == "raw"
    assert column_style(["", ""]) == "raw"
    assert column_style(["1.5", "2.0"]) == "raw"


def test_integers_follow_the_float_style_of_their_column() -> None:
    assert mirror_style("129", "float0") == "129.0"
    assert mirror_style("-35", "float0") == "-35.0"
    assert mirror_style("matched", "float0") == "matched"
    assert mirror_style("129", "raw") == "129"
    assert mirror_style("129", None) == "129"


# --- planning --------------------------------------------------------------
FRESH_COLUMNS = [
    "mri_day", "scanner", "clinical_day", "clinical_match_valid", "supervised_label",
    "training_eligible", "training_exclusion_reason", "dx1", "label_status",
]


def test_only_empty_cells_are_filled_and_disagreements_are_kept() -> None:
    v2 = {"mri_day": "", "scanner": "3.0T", "clinical_day": "", MASTER_FLAG: "True"}
    new = {"mri_day": "129", "scanner": "1.5T", "clinical_day": "0"}
    plan = plan_cells(CAT_C, v2, new, FRESH_COLUMNS, {})
    assert plan.updates == {"mri_day": "129", "clinical_day": "0"}
    assert plan.conflicts == ["scanner"]  # kept, not overwritten


def test_category_d_never_receives_a_supervised_label() -> None:
    assert "supervised_label" not in allowed_columns(CAT_D, FRESH_COLUMNS)
    assert "supervised_label" in allowed_columns(CAT_C, FRESH_COLUMNS)


def test_category_b_receives_pairing_but_no_label_and_no_diagnosis() -> None:
    columns = allowed_columns(CAT_B, FRESH_COLUMNS)
    assert "supervised_label" not in columns
    assert "dx1" not in columns and "label_status" not in columns
    assert {"clinical_day", "mri_day", "training_eligible", "training_exclusion_reason"} <= set(columns)


def test_category_a_gets_neither_label_nor_eligibility() -> None:
    columns = allowed_columns(CAT_A, FRESH_COLUMNS)
    assert "training_eligible" not in columns and "supervised_label" not in columns


def test_master_flag_false_becomes_true_but_other_cells_are_never_overwritten() -> None:
    v2 = {"mri_day": "", MASTER_FLAG: "False"}
    plan = plan_cells(CAT_C, v2, {"mri_day": "129"}, FRESH_COLUMNS, {})
    assert plan.updates[MASTER_FLAG] == "True"


def test_filled_integers_mirror_the_destination_style() -> None:
    v2 = {"clinical_day": ""}
    plan = plan_cells(CAT_C, v2, {"clinical_day": "722"}, FRESH_COLUMNS, {"clinical_day": "float0"})
    assert plan.updates["clinical_day"] == "722.0"


# --- counts ----------------------------------------------------------------
def row(subject: str, label: str, eligible: str = "True") -> dict[str, str]:
    return {"subject_id": subject, "session_id": f"{subject}_MR_d0001{label}",
            "supervised_label": label, "training_eligible": eligible}


def test_counts_by_session_and_by_subject() -> None:
    rows = [row("S1", "CN"), row("S1", "CN"), row("S2", "AD"), row("S3", "", "")]
    counts = compute_counts(rows)
    assert counts["rows"] == 4 and counts["labelled_rows"] == 3 and counts["unlabelled_rows"] == 1
    assert counts["eligible_sessions"] == {"CN": 2, "AD": 1}
    assert counts["eligible_subjects_by_class"] == {"CN": 1, "AD": 1}


def test_subjects_with_more_than_one_class_are_listed() -> None:
    rows = [row("S1", "CN"), row("S1", "MCI"), row("S2", "AD")]
    assert compute_counts(rows)["multi_class_eligible"] == {"S1": ("CN", "MCI")}


def test_ineligible_rows_are_excluded_from_the_eligible_view_only() -> None:
    rows = [row("S1", "OTHER_DEMENTIA", "False"), row("S2", "CN")]
    counts = compute_counts(rows)
    assert counts["labelled_sessions"] == {"OTHER_DEMENTIA": 1, "CN": 1}
    assert counts["eligible_sessions"] == {"CN": 1}


# --- verification ----------------------------------------------------------
HEADER = ["session_id", "subject_id", "supervised_label", "mri_day", "original_x",
          "clinical_match_valid", "clinical_mri_abs_gap_days", MASTER_FLAG]


def v2_rows() -> list[dict[str, str]]:
    return [
        {"session_id": "S1_MR_d0001", "subject_id": "S1", "supervised_label": "CN", "mri_day": "1.0",
         "original_x": "0.5", "clinical_match_valid": "True", "clinical_mri_abs_gap_days": "10.0",
         MASTER_FLAG: "True"},
        {"session_id": "S2_MR_d0002", "subject_id": "S2", "supervised_label": "", "mri_day": "",
         "original_x": "0.7", "clinical_match_valid": "", "clinical_mri_abs_gap_days": "",
         MASTER_FLAG: ""},
    ]


def with_note(rows: list[dict[str, str]], notes: list[str]) -> list[dict[str, str]]:
    return [{**r, NOTE_COLUMN: n} for r, n in zip(rows, notes)]


def test_a_clean_backfill_passes_verification() -> None:
    new = with_note(v2_rows(), ["", "backfill_v1:C"])
    new[1].update(supervised_label="CN", mri_day="2.0", clinical_match_valid="True",
                  clinical_mri_abs_gap_days="30.0", **{MASTER_FLAG: "True"})
    result = verify_backfill(HEADER, v2_rows(), HEADER + [NOTE_COLUMN], new, WINDOW)
    assert result["regression_rows_identical"] == 1


def test_changing_an_already_labelled_row_fails() -> None:
    new = with_note(v2_rows(), ["", ""])
    new[0]["supervised_label"] = "AD"
    with pytest.raises(BackfillError, match="labelled row"):
        verify_backfill(HEADER, v2_rows(), HEADER + [NOTE_COLUMN], new, WINDOW)


def test_changing_a_radiomic_column_fails() -> None:
    new = with_note(v2_rows(), ["", "backfill_v1:C"])
    new[1]["original_x"] = "9.9"
    with pytest.raises(BackfillError, match="protected column"):
        verify_backfill(HEADER, v2_rows(), HEADER + [NOTE_COLUMN], new, WINDOW)


def test_overwriting_a_non_empty_cell_fails() -> None:
    rows = v2_rows()
    rows[1]["mri_day"] = "5.0"
    new = with_note(rows, ["", "backfill_v1:C"])
    new[1]["mri_day"] = "6.0"
    with pytest.raises(BackfillError, match="overwritten"):
        verify_backfill(HEADER, rows, HEADER + [NOTE_COLUMN], new, WINDOW)


def test_a_label_outside_the_window_fails() -> None:
    new = with_note(v2_rows(), ["", "backfill_v1:C"])
    new[1].update(supervised_label="CN", clinical_match_valid="True", clinical_mri_abs_gap_days="190.0")
    with pytest.raises(BackfillError, match="outside the clinical window"):
        verify_backfill(HEADER, v2_rows(), HEADER + [NOTE_COLUMN], new, WINDOW)


def test_a_label_on_an_invalid_pairing_fails() -> None:
    new = with_note(v2_rows(), ["", "backfill_v1:C"])
    new[1].update(supervised_label="CN", clinical_match_valid="False", clinical_mri_abs_gap_days="10.0")
    with pytest.raises(BackfillError, match="outside the clinical window"):
        verify_backfill(HEADER, v2_rows(), HEADER + [NOTE_COLUMN], new, WINDOW)


def test_a_change_without_a_note_fails() -> None:
    new = with_note(v2_rows(), ["", ""])
    new[1]["mri_day"] = "2.0"
    with pytest.raises(BackfillError, match="note/change mismatch"):
        verify_backfill(HEADER, v2_rows(), HEADER + [NOTE_COLUMN], new, WINDOW)


def test_removed_columns_fail() -> None:
    new = with_note(v2_rows(), ["", ""])
    with pytest.raises(BackfillError, match="columns"):
        verify_backfill(HEADER, v2_rows(), HEADER[:-1] + [NOTE_COLUMN], new, WINDOW)


# --- csv fidelity ----------------------------------------------------------
def test_roundtrip_is_lossless_for_a_pandas_style_file(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_bytes(b'a,b,c\n1,"x, y",\n2.0,"q ""z""",3\n')
    assert_lossless_roundtrip(path)
    header, rows = read_text_table(path)
    assert serialise_table(header, rows) == path.read_bytes()


def test_roundtrip_detects_a_file_it_cannot_reproduce(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_bytes(b"a,b\r\n1,2\r\n")  # CRLF would be silently rewritten as LF
    with pytest.raises(BackfillError):
        assert_lossless_roundtrip(path)


# --- conflict description --------------------------------------------------
def test_numeric_conflicts_are_summarised_without_listing_rows() -> None:
    from scripts.backfill_clinical_labels import describe_conflicts

    text = describe_conflicts([("65.4", "65"), ("70.0", "73"), ("80.5", "80")])
    assert "median 0.50" in text and "max 3.00" in text


def test_categorical_conflicts_are_counted_by_pair() -> None:
    from scripts.backfill_clinical_labels import describe_conflicts

    text = describe_conflicts([("3T", "3.0T"), ("3T", "3.0T"), ("UNKNOWN", "AD")])
    assert "'3T' -> '3.0T' x2" in text and "'UNKNOWN' -> 'AD' x1" in text


def test_conflicting_values_are_recorded_with_the_plan() -> None:
    v2 = {"scanner": "3T", "mri_day": ""}
    plan = plan_cells(CAT_C, v2, {"scanner": "3.0T", "mri_day": "5"}, FRESH_COLUMNS, {})
    assert plan.conflict_values == [("scanner", "3T", "3.0T")]
