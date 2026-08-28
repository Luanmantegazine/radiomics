"""Tests for clinical trajectories and the anti-leakage guarantee."""

from __future__ import annotations

import pytest

from oasis_radiomics.clinical.models import (
    ClinicalClassification,
    ClinicalMatch,
    ClinicalVisit,
)
from oasis_radiomics.clinical.trajectories import (
    annotate_session,
    build_trajectories,
    build_trajectory,
    conversion_between,
    diagnosis_label,
)


def _classification(status: str, etiology: str = "UNKNOWN") -> ClinicalClassification:
    return ClinicalClassification(
        cognitive_status=status, ad_etiology=etiology, status="classified"
    )


def _history(subject: str, days_and_status):
    """Build ``(visits, classifications)`` from ``[(day, status, etiology), ...]``."""
    visits = [ClinicalVisit(subject, day) for day, _, _ in days_and_status]
    classifications = {
        day: _classification(status, etiology) for day, status, etiology in days_and_status
    }
    return visits, classifications


CN_MCI_AD = [(500, "CN", "NON_AD"), (1000, "MCI", "UNCERTAIN"), (2000, "DEMENTIA", "AD")]


def _match(subject: str, mri_day: int, visit_day: int) -> ClinicalMatch:
    return ClinicalMatch(
        session_id=f"{subject}_MR_d{mri_day:04d}",
        subject_id=subject,
        mri_day=mri_day,
        visit=ClinicalVisit(subject, visit_day),
        gap_days=visit_day - mri_day,
        abs_gap_days=abs(visit_day - mri_day),
        reason="matched",
        valid=True,
    )


# --- labelling -------------------------------------------------------------
def test_diagnosis_label_marks_ad_dementia() -> None:
    assert diagnosis_label("DEMENTIA", "AD") == "AD"


def test_diagnosis_label_keeps_non_ad_dementia_generic() -> None:
    assert diagnosis_label("DEMENTIA", "NON_AD") == "DEMENTIA"


def test_diagnosis_label_passes_through_other_statuses() -> None:
    assert diagnosis_label("MCI", "AD") == "MCI"
    assert diagnosis_label("CN", "UNKNOWN") == "CN"
    assert diagnosis_label("UNKNOWN", "UNKNOWN") == "UNKNOWN"


# --- ordering --------------------------------------------------------------
def test_trajectory_is_ordered_by_day_not_input_order() -> None:
    visits, classifications = _history("S", CN_MCI_AD)
    trajectory, _ = build_trajectory("S", list(reversed(visits)), classifications)
    assert [point.clinical_day for point in trajectory.points] == [500, 1000, 2000]
    assert trajectory.trajectory == "CN -> MCI -> AD"


def test_trajectory_collapses_repeated_labels() -> None:
    visits, classifications = _history(
        "S", [(100, "CN", "NON_AD"), (400, "CN", "NON_AD"), (900, "MCI", "UNCERTAIN")]
    )
    trajectory, _ = build_trajectory("S", visits, classifications)
    assert trajectory.trajectory == "CN -> MCI"
    assert trajectory.n_clinical_visits == 3


def test_trajectory_endpoints_and_followup() -> None:
    visits, classifications = _history("S", CN_MCI_AD)
    trajectory, _ = build_trajectory("S", visits, classifications)
    assert trajectory.baseline_diagnosis == "CN"
    assert trajectory.last_diagnosis == "AD"
    assert trajectory.followup_years == pytest.approx(1500 / 365.25)


def test_stable_subject_has_no_conversion() -> None:
    visits, classifications = _history(
        "S", [(100, "MCI", "UNCERTAIN"), (900, "MCI", "UNCERTAIN")]
    )
    trajectory, _ = build_trajectory("S", visits, classifications)
    assert trajectory.conversions == ()
    assert trajectory.conversion_event is None


def test_all_conversions_are_tracked() -> None:
    """A subject can convert twice; keeping only the first would hide the second."""
    visits, classifications = _history("S", CN_MCI_AD)
    trajectory, _ = build_trajectory("S", visits, classifications)
    assert trajectory.conversions == (("CN_to_MCI", 1000), ("MCI_to_AD", 2000))


def test_unknown_visit_does_not_fabricate_a_conversion() -> None:
    visits, classifications = _history(
        "S", [(100, "MCI", "UNCERTAIN"), (500, "UNKNOWN", "UNKNOWN"), (900, "MCI", "UNCERTAIN")]
    )
    trajectory, _ = build_trajectory("S", visits, classifications)
    assert trajectory.conversions == ()


def test_non_monotonic_history_is_flagged_not_corrected() -> None:
    visits, classifications = _history(
        "S", [(100, "DEMENTIA", "AD"), (900, "CN", "NON_AD")]
    )
    trajectory, issues = build_trajectory("S", visits, classifications)
    assert any(issue.code == "non_monotonic_trajectory" for issue in issues)
    assert trajectory.last_diagnosis == "CN"  # reported as recorded


def test_next_conversion_after() -> None:
    visits, classifications = _history("S", CN_MCI_AD)
    trajectory, _ = build_trajectory("S", visits, classifications)
    assert trajectory.next_conversion_after(0) == ("CN_to_MCI", 1000)
    assert trajectory.next_conversion_after(1000) == ("MCI_to_AD", 2000)
    assert trajectory.next_conversion_after(2000) is None


# --- the leakage guarantee -------------------------------------------------
def test_future_ad_never_relabels_an_mci_scan() -> None:
    """The core requirement: an MRI keeps the diagnosis it had at scan time."""
    visits, classifications = _history("S", CN_MCI_AD)
    trajectory, _ = build_trajectory("S", visits, classifications)

    row = annotate_session(_match("S", 1200, 1000), classifications[1000], trajectory)

    assert row["diagnosis_at_mri"] == "MCI"
    assert row["future_diagnosis"] == "AD"
    assert row["conversion_event"] == "MCI_to_AD"
    assert row["conversion_day"] == 2000
    assert row["days_to_conversion"] == 800


def test_specification_example_is_reproduced() -> None:
    """clinical 500/1200 = MCI, 2000 = AD; MRI at 800 stays MCI."""
    visits, classifications = _history(
        "S",
        [(500, "MCI", "UNCERTAIN"), (1200, "MCI", "UNCERTAIN"), (2000, "DEMENTIA", "AD")],
    )
    trajectory, _ = build_trajectory("S", visits, classifications)
    row = annotate_session(_match("S", 800, 1200), classifications[1200], trajectory)

    assert row["diagnosis_at_mri"] == "MCI"
    assert row["future_diagnosis"] == "AD"
    assert row["conversion_event"] == "MCI_to_AD"
    assert row["conversion_day"] == 2000
    assert row["days_to_conversion"] == 1200


def test_baseline_scan_keeps_cn_despite_later_dementia() -> None:
    visits, classifications = _history("S", CN_MCI_AD)
    trajectory, _ = build_trajectory("S", visits, classifications)
    row = annotate_session(_match("S", 600, 500), classifications[500], trajectory)
    assert row["diagnosis_at_mri"] == "CN"
    assert row["conversion_event"] == "CN_to_MCI"
    assert row["days_to_conversion"] == 400


def test_past_conversion_is_not_reported_as_future() -> None:
    visits, classifications = _history("S", CN_MCI_AD)
    trajectory, _ = build_trajectory("S", visits, classifications)
    row = annotate_session(_match("S", 2100, 2000), classifications[2000], trajectory)
    assert row["diagnosis_at_mri"] == "AD"
    assert row["conversion_event"] is None
    assert row["days_to_conversion"] is None
    assert row["future_diagnosis"] is None


def test_session_without_a_clinical_match_is_unknown() -> None:
    row = annotate_session(
        ClinicalMatch("S_MR_d0100", "S", 100, reason="no_clinical_visit"), None, None
    )
    assert row["diagnosis_at_mri"] == "UNKNOWN"
    assert row["conversion_event"] is None


# --- interval conversions --------------------------------------------------
def test_conversion_between_visits() -> None:
    visits, classifications = _history("S", CN_MCI_AD)
    trajectory, _ = build_trajectory("S", visits, classifications)
    assert conversion_between(trajectory, 500, 1500) == "CN_to_MCI"
    assert conversion_between(trajectory, 1500, 2500) == "MCI_to_AD"
    assert conversion_between(trajectory, 2000, 2500) is None
    assert conversion_between(None, 0, 9999) is None


def test_build_trajectories_covers_every_subject() -> None:
    visits_s, classifications_s = _history("S", CN_MCI_AD)
    visits_t, classifications_t = _history("T", [(10, "CN", "NON_AD")])
    trajectories, _ = build_trajectories(
        {"S": visits_s, "T": visits_t}, {"S": classifications_s, "T": classifications_t}
    )
    assert set(trajectories) == {"S", "T"}
    assert trajectories["T"].conversion_event is None
