"""Checks for the one-session-per-subject CN/AD cohort filter."""

from pathlib import Path

import pandas as pd
import pytest

from oasis_analysis.filter_supervised_labels import filter_csv


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def row(subject: str, day: int, label: str, gap: int, eligible: object = True) -> dict[str, object]:
    return {
        "subject_id": subject,
        "session_id": f"{subject}_MR_d{day:04d}",
        "age_at_mri": 70,
        "sex": "F",
        "supervised_label": label,
        "clinical_mri_gap_days": gap,
        "training_eligible": eligible,
    }


def run(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[pd.DataFrame, dict[str, int]]:
    source = tmp_path / "input.csv"
    output = tmp_path / "kept.csv"
    write_rows(source, rows)
    stats = filter_csv(source, output, tmp_path / "simplified.csv", tmp_path / "discarded.csv")
    return pd.read_csv(output), stats


def test_ineligible_sessions_are_never_selected(tmp_path: Path) -> None:
    kept, stats = run(tmp_path, [
        row("OAS1", 0, "CN", 2, eligible=False),
        row("OAS1", 500, "CN", 90),
        row("OAS2", 0, "AD", 700, eligible="False"),
        row("OAS3", 0, "AD", 10),
    ])

    assert kept["session_id"].tolist() == ["OAS1_MR_d0500", "OAS3_MR_d0000"]
    assert stats["ineligible_rows"] == 2
    assert stats["cn_kept"] == 1 and stats["ad_kept"] == 1


def test_conflicting_labels_exclude_subject_even_if_one_session_is_ineligible(
    tmp_path: Path,
) -> None:
    kept, stats = run(tmp_path, [
        row("OAS1", 0, "CN", 5),
        row("OAS1", 900, "AD", 400, eligible=False),
        row("OAS2", 0, "CN", 5),
        row("OAS3", 0, "AD", 5),
    ])

    assert set(kept["subject_id"]) == {"OAS2", "OAS3"}
    assert stats["mixed_subjects"] == 1


def test_missing_eligibility_column_keeps_previous_behaviour(tmp_path: Path) -> None:
    rows = [row("OAS1", 0, "CN", 50), row("OAS1", 300, "CN", 5), row("OAS2", 0, "AD", 5)]
    for item in rows:
        del item["training_eligible"]
    kept, stats = run(tmp_path, rows)

    assert kept["session_id"].tolist() == ["OAS1_MR_d0300", "OAS2_MR_d0000"]
    assert stats["ineligible_rows"] == 0


def test_unrecognized_eligibility_values_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unrecognized"):
        run(tmp_path, [row("OAS1", 0, "CN", 5, eligible="maybe"), row("OAS2", 0, "AD", 5)])
