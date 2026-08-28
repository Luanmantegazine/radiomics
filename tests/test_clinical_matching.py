"""Tests for MRI <-> clinical temporal matching and the D1/B4 merge."""

from __future__ import annotations

import pytest

from oasis_radiomics.clinical.matching import (
    MatchingError,
    gap_statistics,
    match_mri_to_clinical,
    match_mri_to_cognitive,
    merge_d1_b4,
    select_nearest,
)
from oasis_radiomics.clinical.models import ClinicalRecord, ClinicalVisit, MriSession


def _mri(subject: str, day: int) -> MriSession:
    return MriSession(
        subject_id=subject, session_id=f"{subject}_MR_d{day:04d}", mri_day=day
    )


def _visit(subject: str, day: int) -> ClinicalVisit:
    return ClinicalVisit(
        subject_id=subject, clinical_day=day, d1_session_id=f"{subject}_UDSd1_d{day:04d}"
    )


def _record(subject: str, day: int, source: str = "d1", **raw) -> ClinicalRecord:
    return ClinicalRecord(
        subject_id=subject,
        clinical_session_id=f"{subject}_UDS{source}_d{day:04d}",
        clinical_day=day,
        source=source,
        raw=dict(raw),
    )


# --- select_nearest --------------------------------------------------------
def test_select_nearest_exact_hit() -> None:
    index, gap, ambiguous = select_nearest(1000, [1000])
    assert (index, gap, ambiguous) == (0, 0, False)


def test_select_nearest_picks_closest_not_earliest() -> None:
    """MRI 1000 with visits at 900 and 1030 must choose 1030 (gap +30)."""
    index, gap, ambiguous = select_nearest(1000, [900, 1030])
    assert index == 1
    assert gap == 30
    assert ambiguous is False


def test_select_nearest_gap_sign_encodes_direction() -> None:
    assert select_nearest(1000, [900])[1] == -100   # visit before the scan
    assert select_nearest(1000, [1100])[1] == 100   # visit after the scan


def test_select_nearest_tie_prefers_earlier_and_flags() -> None:
    """Equal distance must be deterministic and reported, never hidden."""
    index, gap, ambiguous = select_nearest(1000, [900, 1100])
    assert index == 0
    assert gap == -100
    assert ambiguous is True


def test_select_nearest_tie_is_order_independent() -> None:
    assert select_nearest(1000, [1100, 900]) == (1, -100, True)
    assert select_nearest(1000, [900, 1100]) == (0, -100, True)


def test_select_nearest_requires_candidates() -> None:
    with pytest.raises(MatchingError):
        select_nearest(1000, [])


# --- match_mri_to_clinical -------------------------------------------------
def test_exact_match_is_valid_with_zero_gap() -> None:
    match = match_mri_to_clinical([_mri("OAS30001", 1000)], [_visit("OAS30001", 1000)], 180)[0]
    assert match.gap_days == 0
    assert match.abs_gap_days == 0
    assert match.valid is True
    assert match.reason == "matched"


def test_nearest_match_within_window() -> None:
    match = match_mri_to_clinical(
        [_mri("OAS30001", 1000)],
        [_visit("OAS30001", 900), _visit("OAS30001", 1030)],
        180,
    )[0]
    assert match.visit.clinical_day == 1030
    assert match.gap_days == 30
    assert match.valid is True


def test_outside_window_is_kept_and_flagged() -> None:
    """A distant visit must never make the MRI session disappear."""
    match = match_mri_to_clinical([_mri("OAS30001", 1000)], [_visit("OAS30001", 1300)], 180)[0]
    assert match.found is True
    assert match.valid is False
    assert match.reason == "outside_window"
    assert match.abs_gap_days == 300


def test_window_boundary_is_inclusive() -> None:
    assert match_mri_to_clinical([_mri("S", 1000)], [_visit("S", 1180)], 180)[0].valid is True
    assert match_mri_to_clinical([_mri("S", 1000)], [_visit("S", 1181)], 180)[0].valid is False


def test_different_subject_never_matches() -> None:
    """Subjects are never merged across OASIS ids."""
    match = match_mri_to_clinical([_mri("OAS30001", 1000)], [_visit("OAS30002", 1000)], 180)[0]
    assert match.found is False
    assert match.reason == "no_clinical_visit"
    assert match.valid is False
    assert match.candidates_considered == 0


def test_subject_without_any_clinical_visit_is_reported() -> None:
    match = match_mri_to_clinical([_mri("OAS30001", 1000)], [], 180)[0]
    assert match.reason == "no_clinical_visit"


def test_ambiguous_match_is_flagged_but_still_usable() -> None:
    match = match_mri_to_clinical(
        [_mri("S", 1000)], [_visit("S", 900), _visit("S", 1100)], 180
    )[0]
    assert match.ambiguous is True
    assert match.reason == "ambiguous_equal_distance"
    assert match.visit.clinical_day == 900
    assert match.valid is True


def test_every_mri_session_produces_exactly_one_match() -> None:
    sessions = [_mri("S", 100), _mri("S", 900), _mri("T", 50)]
    matches = match_mri_to_clinical(sessions, [_visit("S", 120)], 180)
    assert len(matches) == len(sessions)
    assert [match.session_id for match in matches] == [s.session_id for s in sessions]


def test_negative_window_is_rejected() -> None:
    with pytest.raises(MatchingError):
        match_mri_to_clinical([_mri("S", 1)], [_visit("S", 1)], -1)


def test_configurable_window_changes_validity() -> None:
    sessions, visits = [_mri("S", 1000)], [_visit("S", 1300)]
    assert match_mri_to_clinical(sessions, visits, 180)[0].valid is False
    assert match_mri_to_clinical(sessions, visits, 365)[0].valid is True


# --- cognitive (C1) matching ----------------------------------------------
def test_cognitive_match_is_independent_of_the_diagnostic_visit() -> None:
    """C1 keeps its own day and gap; it is not forced onto the D1/B4 day."""
    matches = match_mri_to_cognitive(
        [_mri("S", 1000)], [_record("S", 1014, source="c1")], 180
    )
    assert matches[0].record.clinical_day == 1014
    assert matches[0].gap_days == 14
    assert matches[0].valid is True


def test_cognitive_match_absent_is_reported() -> None:
    matches = match_mri_to_cognitive([_mri("S", 1000)], [], 180)
    assert matches[0].valid is False
    assert matches[0].reason == "no_clinical_visit"


# --- D1 / B4 merge ---------------------------------------------------------
def test_merge_pairs_on_subject_and_day() -> None:
    visits, issues = merge_d1_b4(
        [_record("S", 100, "d1", NORMCOG="1")], [_record("S", 100, "b4", CDRTOT="0")]
    )
    assert len(visits) == 1
    assert visits[0].has_d1 and visits[0].has_b4
    assert visits[0].source == "d1+b4"
    assert visits[0].d1_value("NORMCOG") == "1"
    assert visits[0].b4_value("CDRTOT") == "0"


def test_missing_b4_does_not_remove_the_d1_diagnosis() -> None:
    visits, issues = merge_d1_b4([_record("S", 100, "d1", NORMCOG="1")], [])
    assert len(visits) == 1
    assert visits[0].has_d1 and not visits[0].has_b4
    assert visits[0].d1_value("NORMCOG") == "1"
    assert any(issue.code == "missing_b4" for issue in issues)


def test_missing_d1_does_not_remove_the_b4_record() -> None:
    visits, issues = merge_d1_b4([], [_record("S", 100, "b4", CDRTOT="1")])
    assert len(visits) == 1
    assert visits[0].has_b4 and not visits[0].has_d1
    assert any(issue.code == "missing_d1" for issue in issues)


def test_duplicate_visits_are_reported_not_silently_resolved() -> None:
    duplicate = ClinicalRecord("S", "S_UDSd1_d0100_b", 100, "d1", raw={"NORMCOG": "0"})
    visits, issues = merge_d1_b4([_record("S", 100, "d1", NORMCOG="1"), duplicate], [])
    assert len(visits) == 1
    assert any(issue.code == "duplicate_clinical_visit" for issue in issues)


def test_merge_is_deterministic_regardless_of_input_order() -> None:
    a = _record("S", 100, "d1", NORMCOG="1")
    b = ClinicalRecord("S", "S_UDSd1_d0100_b", 100, "d1", raw={"NORMCOG": "0"})
    first, _ = merge_d1_b4([a, b], [])
    second, _ = merge_d1_b4([b, a], [])
    assert first[0].d1_value("NORMCOG") == second[0].d1_value("NORMCOG")


def test_merged_visits_are_sorted_by_subject_then_day() -> None:
    visits, _ = merge_d1_b4(
        [_record("T", 50, "d1"), _record("S", 900, "d1"), _record("S", 100, "d1")], []
    )
    assert [(v.subject_id, v.clinical_day) for v in visits] == [
        ("S", 100),
        ("S", 900),
        ("T", 50),
    ]


# --- gap statistics --------------------------------------------------------
def test_gap_statistics_on_empty_input() -> None:
    assert gap_statistics([]) == {"median_gap_days": None, "p95_gap_days": None}


def test_gap_statistics_median() -> None:
    matches = match_mri_to_clinical(
        [_mri("S", 100), _mri("S", 200)], [_visit("S", 100), _visit("S", 210)], 180
    )
    stats = gap_statistics(matches)
    assert stats["median_gap_days"] == pytest.approx(5.0)
