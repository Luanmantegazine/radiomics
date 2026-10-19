"""Merging of clinical instruments and temporal MRI <-> clinical linkage.

Two independent steps:

1. :func:`merge_d1_b4` builds one clinical visit per ``(subject, day)`` from the
   diagnostic (D1) and CDR (B4) instruments. Neither instrument can erase the
   other: a D1 diagnosis with no B4 row survives, and vice versa.
2. :func:`match_mri_to_clinical` links each MRI session to the *nearest in time*
   clinical visit of the **same subject**, within a configurable window.

Matching rules (frozen; see ``CLINICAL_LINKAGE_PROTOCOL.md``)
------------------------------------------------------------
* Candidates are restricted to the same ``subject_id``. Subjects are never
  merged across OASIS ids.
* The selected visit minimises ``abs(clinical_day - mri_day)``.
* **Ties** (two visits equidistant) are resolved deterministically in favour of
  the **earlier** visit, and flagged. Preferring the past is the conservative
  choice: it never lets a later assessment describe an earlier scan.
* A session whose nearest visit falls outside the window is **kept** with
  ``clinical_match_valid = False`` and ``reason = 'outside_window'``. Nothing is
  ever silently dropped.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

from . import DEFAULT_CLINICAL_WINDOW_DAYS
from .models import (
    ClinicalMatch,
    ClinicalRecord,
    ClinicalVisit,
    CognitiveMatch,
    MriSession,
)
from .validation import ValidationIssue

logger = logging.getLogger(__name__)

REASON_MATCHED = "matched"
REASON_OUTSIDE_WINDOW = "outside_window"
REASON_NO_VISIT = "no_clinical_visit"
REASON_AMBIGUOUS = "ambiguous_equal_distance"


class MatchingError(ValueError):
    """Raised when matching is asked to run with impossible parameters."""


# ---------------------------------------------------------------------------
# D1 + B4 merge
# ---------------------------------------------------------------------------
def _index_by_subject_day(
    records: Sequence[ClinicalRecord], issues: list[ValidationIssue]
) -> dict[tuple[str, int], ClinicalRecord]:
    """Index records by ``(subject, day)``, reporting duplicates.

    When a subject has two rows of the same instrument on the same day the
    record with the lexicographically smallest session id is kept, so the result
    is deterministic, and a ``duplicate_clinical_visit`` issue is raised naming
    both.
    """
    indexed: dict[tuple[str, int], ClinicalRecord] = {}
    for record in sorted(
        records, key=lambda item: (item.subject_id, item.clinical_day, item.clinical_session_id)
    ):
        key = (record.subject_id, record.clinical_day)
        existing = indexed.get(key)
        if existing is None:
            indexed[key] = record
            continue
        issues.append(
            ValidationIssue(
                code="duplicate_clinical_visit",
                severity="warning",
                subject_id=record.subject_id,
                identifier=record.clinical_session_id,
                detail=(
                    f"{record.source.upper()} has multiple rows for day "
                    f"{record.clinical_day}: kept {existing.clinical_session_id}, "
                    f"also saw {record.clinical_session_id}."
                ),
            )
        )
    return indexed


def merge_d1_b4(
    d1_records: Sequence[ClinicalRecord],
    b4_records: Sequence[ClinicalRecord],
) -> tuple[list[ClinicalVisit], list[ValidationIssue]]:
    """Merge D1 and B4 on ``OASISID`` + ``days_to_visit``.

    Returns
    -------
    tuple
        ``(visits, issues)``. Visits are sorted by subject and then by day.
        Issues record duplicate rows and visits covered by only one instrument.
    """
    issues: list[ValidationIssue] = []
    d1_index = _index_by_subject_day(d1_records, issues)
    b4_index = _index_by_subject_day(b4_records, issues)

    visits: list[ClinicalVisit] = []
    for key in sorted(set(d1_index) | set(b4_index)):
        subject_id, day = key
        d1 = d1_index.get(key)
        b4 = b4_index.get(key)

        if d1 is None:
            issues.append(
                ValidationIssue(
                    code="missing_d1",
                    severity="info",
                    subject_id=subject_id,
                    identifier=b4.clinical_session_id if b4 else None,
                    detail=f"Day {day} has B4 but no D1 diagnosis row.",
                )
            )
        if b4 is None:
            issues.append(
                ValidationIssue(
                    code="missing_b4",
                    severity="info",
                    subject_id=subject_id,
                    identifier=d1.clinical_session_id if d1 else None,
                    detail=f"Day {day} has D1 but no B4 CDR row.",
                )
            )

        visits.append(
            ClinicalVisit(
                subject_id=subject_id,
                clinical_day=day,
                d1_session_id=d1.clinical_session_id if d1 else None,
                b4_session_id=b4.clinical_session_id if b4 else None,
                age_at_clinical_visit=_first_age(d1, b4),
                d1=d1,
                b4=b4,
            )
        )

    both = sum(1 for visit in visits if visit.has_d1 and visit.has_b4)
    logger.info(
        "Merged clinical visits: %d total (%d with D1+B4, %d D1-only, %d B4-only).",
        len(visits),
        both,
        sum(1 for visit in visits if visit.has_d1 and not visit.has_b4),
        sum(1 for visit in visits if visit.has_b4 and not visit.has_d1),
    )
    return visits, issues


def _first_age(d1: ClinicalRecord | None, b4: ClinicalRecord | None) -> float | None:
    """Age at visit, preferring D1 (the primary diagnostic source)."""
    for record in (d1, b4):
        if record is not None and record.age_at_visit is not None:
            return record.age_at_visit
    return None


def group_visits_by_subject(
    visits: Sequence[ClinicalVisit],
) -> dict[str, list[ClinicalVisit]]:
    """Group clinical visits per subject, ordered by ``clinical_day``."""
    grouped: dict[str, list[ClinicalVisit]] = {}
    for visit in visits:
        grouped.setdefault(visit.subject_id, []).append(visit)
    for items in grouped.values():
        items.sort(key=lambda item: item.clinical_day)
    return dict(sorted(grouped.items()))


# ---------------------------------------------------------------------------
# nearest-in-time selection (shared by clinical and cognitive matching)
# ---------------------------------------------------------------------------
def select_nearest(mri_day: int, candidate_days: Sequence[int]) -> tuple[int, int, bool]:
    """Pick the candidate day closest to ``mri_day``.

    Parameters
    ----------
    mri_day:
        Day of the MRI session.
    candidate_days:
        Days of the candidate visits, in any order.

    Returns
    -------
    tuple
        ``(index, gap_days, ambiguous)`` where ``gap_days = candidate - mri_day``
        (negative when the visit precedes the scan) and ``ambiguous`` is ``True``
        when at least two candidates share the minimum absolute gap.

    Raises
    ------
    MatchingError
        If ``candidate_days`` is empty; callers must handle "no visit" before
        asking for a nearest one.

    Notes
    -----
    Ties are broken in favour of the **earlier** visit, so the result never
    depends on input ordering and never prefers a future assessment over an
    equally distant past one.
    """
    if not candidate_days:
        raise MatchingError("select_nearest requires at least one candidate day.")

    gaps = [day - mri_day for day in candidate_days]
    minimum = min(abs(gap) for gap in gaps)
    tied = [index for index, gap in enumerate(gaps) if abs(gap) == minimum]

    # Deterministic tie-break: smallest gap value = earliest day.
    chosen = min(tied, key=lambda index: (gaps[index], candidate_days[index]))
    return chosen, gaps[chosen], len(tied) > 1


# ---------------------------------------------------------------------------
# MRI <-> clinical visit
# ---------------------------------------------------------------------------
def match_mri_to_clinical(
    mri_sessions: Sequence[MriSession],
    visits: Sequence[ClinicalVisit],
    window_days: int = DEFAULT_CLINICAL_WINDOW_DAYS,
) -> list[ClinicalMatch]:
    """Link every MRI session to its nearest clinical visit.

    Every input session produces exactly one :class:`ClinicalMatch`, including
    sessions with no clinical data at all and sessions whose nearest visit lies
    outside ``window_days``.
    """
    if window_days < 0:
        raise MatchingError(f"window_days must be >= 0, got {window_days}")

    by_subject = group_visits_by_subject(visits)
    matches: list[ClinicalMatch] = []

    for session in mri_sessions:
        candidates = by_subject.get(session.subject_id, [])
        if not candidates:
            matches.append(
                ClinicalMatch(
                    session_id=session.session_id,
                    subject_id=session.subject_id,
                    mri_day=session.mri_day,
                    reason=REASON_NO_VISIT,
                    valid=False,
                    candidates_considered=0,
                )
            )
            continue

        index, gap, ambiguous = select_nearest(
            session.mri_day, [visit.clinical_day for visit in candidates]
        )
        visit = candidates[index]
        within_window = abs(gap) <= window_days

        if not within_window:
            reason = REASON_OUTSIDE_WINDOW
        elif ambiguous:
            reason = REASON_AMBIGUOUS
        else:
            reason = REASON_MATCHED

        matches.append(
            ClinicalMatch(
                session_id=session.session_id,
                subject_id=session.subject_id,
                mri_day=session.mri_day,
                visit=visit,
                gap_days=gap,
                abs_gap_days=abs(gap),
                reason=reason,
                valid=within_window,
                ambiguous=ambiguous,
                candidates_considered=len(candidates),
            )
        )

    _log_match_summary(matches, window_days)
    return matches


def _log_match_summary(matches: Sequence[ClinicalMatch], window_days: int) -> None:
    """Log how the MRI sessions were resolved."""
    valid = sum(1 for match in matches if match.valid)
    outside = sum(1 for match in matches if match.reason == REASON_OUTSIDE_WINDOW)
    none = sum(1 for match in matches if match.reason == REASON_NO_VISIT)
    ambiguous = sum(1 for match in matches if match.ambiguous)
    logger.info(
        "MRI<->clinical (+/-%dd): %d session(s) -> %d valid, %d outside window, "
        "%d without clinical data, %d ambiguous tie(s).",
        window_days,
        len(matches),
        valid,
        outside,
        none,
        ambiguous,
    )


# ---------------------------------------------------------------------------
# MRI <-> cognitive (C1)
# ---------------------------------------------------------------------------
def match_mri_to_cognitive(
    mri_sessions: Sequence[MriSession],
    records: Sequence[ClinicalRecord],
    window_days: int = DEFAULT_CLINICAL_WINDOW_DAYS,
) -> list[CognitiveMatch]:
    """Link every MRI session to its nearest C1 psychometric assessment.

    Deliberately independent of :func:`match_mri_to_clinical`: psychometric
    testing frequently happens a few days off the diagnostic visit, so C1 keeps
    its own day, gap and validity columns.
    """
    if window_days < 0:
        raise MatchingError(f"window_days must be >= 0, got {window_days}")

    by_subject: dict[str, list[ClinicalRecord]] = {}
    for record in records:
        by_subject.setdefault(record.subject_id, []).append(record)
    for items in by_subject.values():
        items.sort(key=lambda item: (item.clinical_day, item.clinical_session_id))

    matches: list[CognitiveMatch] = []
    for session in mri_sessions:
        candidates = by_subject.get(session.subject_id, [])
        if not candidates:
            matches.append(
                CognitiveMatch(session_id=session.session_id, reason=REASON_NO_VISIT)
            )
            continue

        index, gap, ambiguous = select_nearest(
            session.mri_day, [record.clinical_day for record in candidates]
        )
        within_window = abs(gap) <= window_days
        if not within_window:
            reason = REASON_OUTSIDE_WINDOW
        elif ambiguous:
            reason = REASON_AMBIGUOUS
        else:
            reason = REASON_MATCHED

        matches.append(
            CognitiveMatch(
                session_id=session.session_id,
                record=candidates[index],
                gap_days=gap,
                abs_gap_days=abs(gap),
                valid=within_window,
                reason=reason,
            )
        )

    logger.info(
        "MRI<->cognitive (+/-%dd): %d of %d session(s) matched within the window.",
        window_days,
        sum(1 for match in matches if match.valid),
        len(matches),
    )
    return matches


def gap_statistics(matches: Sequence[ClinicalMatch]) -> dict[str, float | None]:
    """Median and 95th percentile of the absolute MRI<->clinical gap."""
    gaps = sorted(
        match.abs_gap_days for match in matches if match.abs_gap_days is not None
    )
    if not gaps:
        return {"median_gap_days": None, "p95_gap_days": None}

    def percentile(fraction: float) -> float:
        position = fraction * (len(gaps) - 1)
        low = int(position)
        high = min(low + 1, len(gaps) - 1)
        weight = position - low
        return gaps[low] * (1 - weight) + gaps[high] * weight

    return {"median_gap_days": percentile(0.5), "p95_gap_days": percentile(0.95)}
